import os
from dotenv import load_dotenv
import openai
import argparse
import json
import tiktoken
from colorama import init, Fore, Style
from datetime import datetime

init(autoreset=True)


class TrainGPT:
    def __init__(self, api_key=None, model_name=None):
        load_dotenv()

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Please set the OPENAI_API_KEY environment variable or provide the API key as an argument.")

        if model_name is None:
            model_name = os.getenv("OPENAI_BASE_MODEL", "gpt-4-0613")

        openai.api_key = api_key
        self.model_name = model_name
        self.file_id = None
        self.job_id = None
        self.model_id = None
        self.file_path = None

    def create_file(self, file_path):
        self.file_path = file_path
        file = openai.File.create(file=open(file_path, "rb"), purpose='fine-tune')
        self.file_id = file.id
        print(f"File ID: {self.file_id}")

    def list_files(self, field='bytes', direction='asc'):
        files = openai.File.list()
        file_data = files['data']

        if field:
            file_data = sorted(file_data, key=lambda x: x[field], reverse=(direction == 'desc'))

        print(f"{Fore.GREEN}{'ID':<30}{'Bytes (MB)':<20}{'Created At'}{Style.RESET_ALL}")
        for file in file_data:
            created_at = datetime.fromtimestamp(file['created_at']).strftime('%Y-%m-%d %H:%M:%S')
            bytes_mb = file['bytes'] / (1024 * 1024)
            print(
                f"{Fore.CYAN}{file['id']:<30}{Fore.YELLOW}{bytes_mb:.2f} MB{Fore.MAGENTA}{created_at:>25}{Style.RESET_ALL}")

    def delete_file(self, file_id=None):
        if file_id is None:
            file_id = self.file_id
            if file_id is None:
                raise ValueError("File not set. Please provide a file ID.")

        openai.File.delete(file_id)
        print(f"File ID: {file_id} deleted.")

    def get_file_details(self, file_id=None):
        if file_id is None:
            file_id = self.file_id
            if file_id is None:
                raise ValueError("File not set. Please upload a file first using the 'create_file' method.")

        file = openai.File.retrieve(file_id)
        print(json.dumps(file, indent=2))
        return file

    def start_training(self, file_id=None):
        if file_id is None:
            file_id = self.file_id
            if file_id is None:
                raise ValueError("File not set. Please upload a file first using the 'create_file' method.")

        job = openai.FineTuningJob.create(training_file=file_id, model=self.model_name)
        self.job_id = job.id
        print(f"Job ID: {self.job_id}")

    def list_jobs(self, limit=10):
        jobs = openai.FineTuningJob.list(limit=limit)

        print(f"{Fore.GREEN}{'ID':<35}{'Model':<15}{'Status':<12}{'Created At'}{Style.RESET_ALL}")
        for job in jobs['data']:
            created_at = datetime.fromtimestamp(job['created_at']).strftime('%Y-%m-%d %H:%M:%S')
            print(
                f"{Fore.CYAN}{job['id']:<35}{Fore.YELLOW}{job['model']:<15}{Fore.MAGENTA}{job['status']:<12}{created_at}{Style.RESET_ALL}")

        return jobs

    def get_job_details(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
            if job_id is None:
                raise ValueError("No training job started. Please start a job first using the 'start_training' method.")

        details = openai.FineTuningJob.retrieve(job_id)
        print(json.dumps(details, indent=2))
        return details

    def cancel_job(self, job_id=None):
        if job_id is None:
            job_id = self.job_id
            if job_id is None:
                raise ValueError("No training job started. Please start a job first using the 'start_training' method.")

        openai.FineTuningJob.cancel(job_id)
        print(f"Job {job_id} canceled.")

    def list_events(self, job_id=None, limit=10):
        if job_id is None:
            job_id = self.job_id
            if job_id is None:
                raise ValueError("No training job started. Please start a job first using the 'start_training' method.")

        events = openai.FineTuningJob.list_events(id=job_id, limit=limit)
        print(json.dumps(events, indent=2))
        return events

    def delete_model(self, model_id=None):
        if model_id is None:
            model_id = self.model_id
            if model_id is None:
                raise ValueError("Model ID not set. Please provide a model ID.")

        openai.Model.delete(model_id)
        print(f"Model {model_id} deleted.")

    def list_models(self):
        models = openai.Model.list()
        return models

    def list_models_summary(self):
        models = openai.Model.list()

        print(f"{Fore.GREEN}Total models: {len(models['data'])}{Style.RESET_ALL}\n")

        owners = {}
        for model in models['data']:
            owner = model['owned_by']
            if owner not in owners:
                owners[owner] = []
            owners[owner].append(model)

        for owner, models in owners.items():
            print(f"{Fore.CYAN}{owner}{Style.RESET_ALL}")
            for model in models:
                print(f"  {Fore.YELLOW}{model['id']}{Style.RESET_ALL}")
            print()

    def list_models_by_owner(self, owner):
        models = openai.Model.list()

        owned_models = [model for model in models['data'] if model['owned_by'] == owner]

        if not owned_models:
            print(f"{Fore.RED}No models found for owner: {owner}{Style.RESET_ALL}")
            return

        print(f"{Fore.GREEN}Models owned by {owner}:{Style.RESET_ALL}")
        for model in owned_models:
            created_at = datetime.fromtimestamp(model['created']).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{Fore.CYAN}{model['id']}{Style.RESET_ALL}")
            print(f"  Created: {created_at}")
            print(f"  Root model: {model['root']}")
            print(f"  Parent model: {model['parent']}\n")

    @staticmethod
    def num_tokens_from_string(string, encoding_name):
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))

    @staticmethod
    def count_tokens_from_messages(encoding_name, messages):
        return sum(TrainGPT.num_tokens_from_string(msg['content'], encoding_name) for msg in messages)

    def count_tokens(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
            if file_path is None:
                raise ValueError("Please provide a file path or upload a file first using 'create_file' method.")

        encodings = {'cl100k_base', 'p50k_base', 'r50k_base'}

        tokens = {enc: 0 for enc in encodings}

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                for enc in encodings:
                    tokens[enc] += TrainGPT.count_tokens_from_messages(enc, data['messages'])

        print(f"{Fore.GREEN}Token counts:{Style.RESET_ALL}")
        for enc, count in tokens.items():
            print(f"{Fore.CYAN}{enc}: {Fore.YELLOW}{count}{Style.RESET_ALL}")

        return tokens


def main():
    parser = argparse.ArgumentParser(description="Command Line Interface for TrainGPT")

    parser.add_argument("--api-key", type=str, help="OpenAI API Key")
    parser.add_argument("--model-name", type=str, help="Base model to fine-tune")
    parser.add_argument("--create-file", type=str, help="Path to the file to be uploaded")
    parser.add_argument("--start-training", action="store_true",
                        help="Start a new training job using the uploaded file")
    parser.add_argument("--list-jobs", action="store_true", help="List all training jobs")
    parser.add_argument("--get-job-details", type=str, help="Get details for a specific job")
    parser.add_argument("--cancel-job", type=str, help="Cancel a specific job")
    parser.add_argument("--list-events", type=str, help="List events for a specific job")
    parser.add_argument("--list-models-summary", action="store_true", help="List models summary, grouped by owner")
    parser.add_argument("--list-models-by-owner", type=str, help="List models owned by a specific user")
    parser.add_argument("--delete-model", type=str, help="Delete a specific model")
    parser.add_argument("--count-tokens", type=str, help="Count tokens in a JSONL file")

    args = parser.parse_args()

    trainer = TrainGPT(api_key=args.api_key, model_name=args.model_name)

    if args.create_file:
        trainer.create_file(args.create_file)

    if args.start_training:
        trainer.start_training()

    if args.list_jobs:
        trainer.list_jobs()

    if args.get_job_details:
        trainer.get_job_details(args.get_job_details)

    if args.cancel_job:
        trainer.cancel_job(args.cancel_job)

    if args.list_events:
        trainer.list_events(args.list_events)

    if args.list_models_summary:
        trainer.list_models_summary()

    if args.list_models_by_owner:
        trainer.list_models_by_owner(args.list_models_by_owner)

    if args.delete_model:
        trainer.delete_model(args.delete_model)

    if args.count_tokens:
        trainer.count_tokens(args.count_tokens)


if __name__ == "__main__":
    main()