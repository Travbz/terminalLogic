fetch('./web/webData/portfolio.json')
  .then(response => response.json())
  .then(data => console.log(data));


  var tbody = d3.select("tbody");
  data.forEach(function(data) {
    console.log(data);
    var row = tbody.append("tr");
    Object.entries(data).forEach(function([key, value]) {
      console.log(key, value);
  
      // Append a cell to the row for each value
      // in the weather report object
      var cell = row.append("td");
      cell.text(value);
    });
  });
  
