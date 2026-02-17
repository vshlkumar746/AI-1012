"""

rm -r mslearn-ai-language -f
git clone https://github.com/microsoftlearning/mslearn-ai-language
cd mslearn-ai-language/Labfiles/02-qna/Python/qna-app
python -m venv labenv
./labenv/bin/Activate.ps1
pip install -r requirements.txt azure-ai-language-questionanswering

In Requirment.txt
python-dotenv

IN. env
AI_SERVICE_ENDPOINT=https://language59220113.cognitiveservices.azure.com/
AI_SERVICE_KEY=7sAbKanPmNeV4BNDIYwwwXq9boK6cafjMRXB3cPXBVDHC9QpUa50JQQJ99CBACYeBjFXJ3w3AAAaACOGn4dU
QA_PROJECT_NAME=LearnFAQ
QA_DEPLOYMENT_NAME=production
"""

from dotenv import load_dotenv
import os

# import namespaces
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.questionanswering import QuestionAnsweringClient


def main():
    try:
        # Get Configuration Settings
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')
        ai_project_name = os.getenv('QA_PROJECT_NAME')
        ai_deployment_name = os.getenv('QA_DEPLOYMENT_NAME')

        # Create client using endpoint and key
        credential = AzureKeyCredential(ai_key)
        ai_client = QuestionAnsweringClient(endpoint=ai_endpoint, credential=credential)


        # Submit a question and display the answer
        user_question = ''
        while True:
            user_question = input('\nQuestion:\n')
            if user_question.lower() == "quit":                
                break
            response = ai_client.get_answers(question=user_question,
                                            project_name=ai_project_name,
                                            deployment_name=ai_deployment_name)
            for candidate in response.answers:
                print(candidate.answer)
                print("Confidence: {}".format(candidate.confidence))
                print("Source: {}".format(candidate.source))



    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    main()
