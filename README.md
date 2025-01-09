I have created a chatbot for the aspirants who are preparing for the upsc exams.I have used a pdf books as dataset and inserted in to the pinecone and used rag pipeline to process this chat.

I have created test.py to store the pdf books into the vectorstore . Here i have used Pinecone .
I have created key.env which contain key of openai api and pinecone api.
I have also created requirements.txt file for the dependency.
I have also created conda enviroment with python==3.10.



WORKFLOW:-
I have standarized and commit the code for better undersanding . The flow is whenever user will ask the question and click on submit button. The question will be sent to the generate_openai_prompt function where it generate a proper question on the basis of previous conversation or context and generated question will be sent to rag pipeline where it deliver the answer .
Sometimes it deliver i don't know beacuse the question might not present in the pdf or dataset that have been inserted.

I have also displayed chathistory on the left hand side.

I have run first test.py before running app.py beacuse i have inserted dataset in the pinecone using test.py.


