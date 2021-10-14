var FTXSocket = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@depth')

FTXSocket.onmessage = function (event) {	
	var message = JSON.parse(event.data); 
    b = message.b
    a = message.a
    console.log(message)
};
let obj = {
    b: [],
    a: [],
    u: [],
    depthUpdate: [],
    s: '',
    buffer: [],
};


//     asks.forEach(function(sizequantA) {
//         console.log(sizequantA)
//         var row = tbody.append("tr")        
//         Object.entries(sizequantA).forEach(function([key, value]) {
//             console.log(key, value);
//             var cell = row.append("td");
//             cell.text(value);

//     })
// });
// };
