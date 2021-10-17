d3.csv("../assets/portfolio.csv").then(function(data, err) {
  if (err) throw err;
    var tbody = d3.select("tbody");
    console.log(data);
    data.forEach(function(tableValue) {
        console.log(tableValue);
        var row = tbody.append("tr");
        Object.entries(tableValue).forEach(function([key, value]) {
            console.log(key, value);
            var cell = row.append("td");
            cell.text(value);
      });
    })
  });
