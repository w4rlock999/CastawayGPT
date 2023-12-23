import { cookies } from 'next/headers'

interface Payload {
  chatMessage : string
}

export async function POST(request: Request) {

    const data : Payload = await request.json()  
    
    var sessionID
    if (cookies().has('sessionID')) {
      sessionID = cookies().get('sessionID')
      console.log(sessionID)
    } else {
      console.log("cannot find sessionID in the cookies")
    }

    if (data.chatMessage != "") {

        console.log("chat: " + data.chatMessage)
        try {          
          var response = await fetch('http://127.0.0.1:5000/chatWithContext', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',        
            },
            body: JSON.stringify({
              sessionID: sessionID.value,
              message: data.chatMessage           
            }),
          });         
        }catch (error) {      
          return new Response("internal server error", { status: 500, statusText: "send chatWIthContext failed"})             
        }
        
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