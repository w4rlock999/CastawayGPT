interface Payload {
    promptSearch : string
}

export async function POST(request: Request) {

    const data : Payload = await request.json()  
    
    if (data.promptSearch != "") {

        console.log("chat: " + data.promptSearch)

        const response = await fetch('http://127.0.0.1:5000/chatWithContext', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',        
            },
            body: JSON.stringify(data.promptSearch),
          });
      
          if (response.ok) {
            const result = await response.json()
            console.log(JSON.stringify(result))
            return new Response(JSON.stringify(result))
          } else {      
            return Response.error()      
          }
        return new Response("chat success!");
    }
}