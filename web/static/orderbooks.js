var FTXSocket = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@depth')

FTXSocket.onmessage = function (event) {	
	var message = JSON.parse(event.data);
    var asks = message.a
    var bids = message.b
	console.log(asks)
};
function loadHTML(asks) {
    var tbody = d3.select("tbody");
    console.log(asks);
    asks.forEach(function(books) {
        console.log(books);
        var row = tbody.append("tr");
        Object.entries(books).forEach(function([key, value]) {
            console.log(key, value);
            var cell = row.append("tr");
            cell.text(value);
        });
    });
  };
loadHTML(FTXSocket);