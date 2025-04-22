# Twilio Programmable Voice Call

Case study of implementing a voice based assistant with twilio.

Twilio offers programmable voice call. Under the hood it works in following step:
- When call is initiated, sends a request to incoming callback handler. This has to be manually registered from twilio [console](https://www.twilio.com/console/projects)

- Then, a websocket connection is established. Websocket object consists of media payload consisting of audio bytes. 


## Table Of Contents:

- [Understanding WebSocket Events](./samples/Socket%20Events/)

- [Creating a Wav File Out Of Audio Bytes From WebSocket](./samples/Decoding%20Audio%20Chunks/)

