# lsvtChatBot

## Overview

Lots of people using LangChain are struggling with working with multiple functions. One common issue is the ability to create a simple chat bot that uses the following:

- Prompt Template 
- Chat History
- Vector DB (Pinecone)

It uses `ConversationalRetrievalChain` for the retrieval of the DB.

## How it works

It uses my Prompt Template (`template`),  the context from the embedding  search and the user's question. Provides an answer. Then, in the "background",  for follow up question, it uses the follow up template (`_template`) to combine the new question with the previous answer, to provide an additional context. That additional context is not sent to chatGPT, all it does is create a new question on its own, which would then be passed to the query, retrieve full context from docs and finally use the query/full context to send to chatGPT.

Example Process: KB about house painting
- User Question 1: "How do I paint a door?" -> Vector/Embeddings search -> Get full context -> Send question with context -> Return Answer 1: "Take a brush out, open can of paint, mask the door frame, etc."
- User Question 2: "Do I need a tool to open it?" -> Background process: Combine Answer 1 with question 2 and create question 2-A: "When painting, what tools can be used to open the can of paint?"
- User Question 2 becomes Question 2-A -> Vector/Embeddings search -> Get full context -> Send question with context -> Return Answer 2: "Just use a screwdriver!"

This may get a little "buggy" with lots of follow up questions but should give a base to use for better coding. Also, it needs a lot of finesse with the prompt engineering based on your context and expected tone of answers, so will need to try variations there to be flexible enough for answers.

## Commnents

- This code is only to help to create your own. It is far from perfect. No support is provided.
- If you find it useful and have comments for suggestions or improvements, please help the community and open an issue in this repository so that it can be shared with others.
- If you add features to it, please also share.
- If you have code to share based on wished improvements below, please also add to an issue.
- As time permit, I will try to update code with any suggestions for improvements.

## Improvements

- Token count was mentioned by someone
- Ability to cache some queries
- Find a way to only add vectors to DB if docs have changed or been added
