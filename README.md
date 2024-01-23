# TubeHub.ai
A Youtube Video Summarizer which uses advanced tools like LangChain and Whisper to summarize the video and answer user questions related to the video.

In the digital era, the abundance of information can be overwhelming, and we often find ourselves scrambling to consume as much content as possible within our limited time. YouTube is a treasure trove of knowledge and entertainment, but it can be challenging to sift through long videos to extract the key takeaways. Worry not, as we've got your back! In this lesson, we will unveil a powerful solution to help you efficiently summarize YouTube videos using two cutting-edge tools: Whisper and LangChain.

First, we download the youtube video we are interested in and transcribe it using Whisper. Then, we’ll proceed by creating summaries using two different approaches:

First we use an existing summarization chain to generate the final summary, which automatically manages embeddings and prompts.
Then, we use another approach more step-by-step to generate a final summary formatted in bullet points, consisting in splitting the transcription into chunks, computing their embeddings, and preparing ad-hoc prompts.
