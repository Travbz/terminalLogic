fetch ("../templates/portfolio2.json")
.then(function(resp) {
    return resp.json();
})
.then(function(data) {
    console.log(data)

function loadHTML(tableData) {
    var tbody = d3.select("tbody");
    console.log(tableData);
    tableData.forEach(function(dataObj) {
        console.log(dataObj);
        var row = tbody.append("tr");
        Object.entries(dataObj).forEach(function([key, value]) {
            console.log(key, value);
            var cell = row.append("td");
            cell.text(value);
        });
    });
  };
  loadHTML(data);
})