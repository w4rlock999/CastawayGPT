interface Payload {
  chatMessage : string
}

export async function POST(request: Request) {

    const data : Payload = await request.json()  
    
    if (data.chatMessage != "") {

        console.log("chat: " + data.chatMessage)

        var response = await fetch('http://127.0.0.1:5000/chatWithContext', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',        
          },
          body: JSON.stringify({
            message: data.chatMessage
          }),
        });
        
        if (response.ok) {
          const result = await response.json()
          console.log(JSON.stringify(result))
          return new Response(JSON.stringify(result))
        } else {  
          console.log("response not ok")    
          const result = await response.json()
          console.log(JSON.stringify(result))
          return Response.error()      
        }
        return new Response("chat success!");
    }
}