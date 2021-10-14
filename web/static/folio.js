fetch('./web/webData/portfolio.json')
  .then(response => response.json())
  .then(data => console.log(data));




// data.forEach(function(Report) {
//     console.log(weatherReport);
//     var row = tbody.append("tr");
//     Object.entries(weatherReport).forEach(function([key, value]) {
//       console.log(key, value);
  
//       // Append a cell to the row for each value
//       // in the weather report object
//       var cell = row.append("td");
//       cell.text(value);
//     });
//   });