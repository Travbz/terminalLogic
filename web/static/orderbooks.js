var FTXSocket = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@depth')

FTXSocket.onmessage = function (event) {	
	var message = JSON.parse(event.data); 
    b = message.b
    a = message.a
    console.log(message)
    b.forEach(function(b) {
        console.log(b)
        var row = tbody.append("tr")        
        Object.entries(b).forEach(function([key, value]) {
            console.log(key, value);
            var cell = row.append("td");
            cell.text(value);

    })
});
};
let obj = {
    b: [],
    a: [],
    u: [],
    depthUpdate: [],
    s: '',
    buffer: [],
};



