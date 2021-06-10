// Creating plots and tables for regression models 
function plots(dataFile) {
    d3.json("Website/Results/" + dataFile + ".json").then(function (data) {

        for (i = 9; i < 15; i++) {
            let id = i.toString()

            let prediction = data[id]['results']['Prediction']
            let x1 = Object.keys(prediction)
            let y1 = Object.values(prediction)

            let actual = data[id]['results']['Actual']
            let x2 = Object.keys(actual)
            let y2 = Object.values(actual)

            var trace1 = { x: x1, y: y1, type: "line", name: "Prediction", line: {color: 'rgb(243, 75, 20)'}};
            var trace2 = { x: x2, y: y2, type: "line", name: "Actual", line: {color: 'rgb(117, 163, 51)'}};

            var plotData = [trace1, trace2];

            var layout = {
                width: 450,
                height: 250,
                legend: { "orientation": "v", x: 1, xanchor: "right", y: 1 },
                margin: { l: 40, r: 0, t: 10, b: 30, pad: 4 }
            }

            Plotly.newPlot("plot" + id, plotData, layout, { responsive: true });

            let outcome = data[id]["outcome"]
            let headers = Object.keys(outcome).slice(1)
            let values = Object.values(outcome).slice(1)


            d3.select("#table" + id).html("")
            var table = d3.select("#table" + id).append('table')
            table.attr('class', 'datagrid')
            var thead = table.append('thead')
            var tbody = table.append('tbody')

            var theadRow = thead.append('tr')
            headers.map(header => {
                var cell = theadRow.append('th')
                cell.html(header)
            })

            var tbodyRow = tbody.append('tr')
            values.map(value => {
                var cell = tbodyRow.append('td')
                cell.html(value)
            })
        }
    })
}


// Creating reports and metrics tables for classification models 
function report(dataFile) {
    d3.json("Website/Results/" + dataFile + ".json").then(function (data) {

        for (i = 9; i < 15; i++) {
            let id = i.toString()

            let classification = data[id]['classification']
            let loss_accuracy = data[id]['metrics']

            d3.select("#report" + id).html("")

            if (classification) {
                var table = d3.select("#report" + id).append('table')
                table.attr('class', 'datagrid')
                table.attr('style', 'margin-bottom: 25px')
                var thead = table.append('thead')
                var tbody = table.append('tbody')

                var theadRow = thead.append('tr')
                thead.append('th').attr("class", "column1").html("")
                thead.append('th').html("precision")
                thead.append('th').html("recall")
                thead.append('th').html("f1-score")
                thead.append('th').html("support")

                var row0 = ["TP >0.35", ...Object.values(classification['0'])]
                var row1 = ["TP <0.35", ...Object.values(classification['1'])]
                var row2 = ["accuracy", "", "", classification["accuracy"], ""]
                var row3 = ["macro avg", ...Object.values(classification['macro avg'])]
                var row4 = ["weighted avg", ...Object.values(classification['weighted avg'])]
                var allRows = [row0, row1, row2, row3, row4]

                allRows.map(row => {
                    var tbodyRow = tbody.append('tr')
                    tbodyRow.append('td').attr("class", "column1").html(row[0])
                    for (j = 1; j < 5; j++) {
                        if (row[j] === '')
                            tbodyRow.append('td').html("&nbsp;")
                        else
                            tbodyRow.append('td').html(Number(row[j]).toFixed(j < 4 ? 2 : 0))
                    }
                })
            }

            if (loss_accuracy) {
                let loss_train = loss_accuracy['Train Loss']
                let x1 = Object.keys(loss_train)
                let y1 = Object.values(loss_train)

                let loss_test = loss_accuracy['Test Loss']
                let x2 = Object.keys(loss_test)
                let y2 = Object.values(loss_test)

                let accuracy_train = loss_accuracy['Train Accuracy']
                let x3 = Object.keys(accuracy_train)
                let y3 = Object.values(accuracy_train)

                let accuracy_test = loss_accuracy['Test Accuracy']
                let x4 = Object.keys(accuracy_test)
                let y4 = Object.values(accuracy_test)

                var trace1 = { x: x1, y: y1, xaxis: 'x1', yaxis: 'y1', type: "line", name: "Train Loss", line: {color: 'rgb(243, 75, 20)'}};
                var trace2 = { x: x2, y: y2, xaxis: 'x1', yaxis: 'y1', type: "line", name: "Test Loss", line:{color:'rgb(117, 163, 51)'}};
                var trace3 = { x: x3, y: y3, xaxis: 'x2', yaxis: 'y2', type: "line", name: "Train Accuracy", line:{color: 'rgb(59, 174, 205)'}};
                var trace4 = { x: x4, y: y4, xaxis: 'x2', yaxis: 'y2', type: "line", name: "Test Accuracy", line:{color:'rgb(99, 34, 48)'}};

                var plotData = [trace1,trace2,trace3,trace4];

                var layout = {
                    grid: {rows: 1, columns: 2, pattern:'independent'},
                    subplots: [['x1', 'y1'], ['x2', 'y2']],
                    width: 450,
                    height: 250,
                    legend: { "orientation": "h", x: 1, xanchor: "right", y: 225, font: {size:9} },
                    margin: { l: 40, r: 0, t: 10, b: 30, pad: 4 }
                }
                Plotly.newPlot("report" + id, plotData, layout, { responsive: true })
            }

            let outcome = data[id]["outcome"]
            let headers = Object.keys(outcome).slice(1)
            let values = Object.values(outcome).slice(1)

            d3.select("#metrics" + id).html("")
            var table = d3.select("#metrics" + id).append('table')
            table.attr('class', 'datagrid')
            var thead = table.append('thead')
            var tbody = table.append('tbody')

            var theadRow = thead.append('tr')
            headers.map(header => {
                var cell = theadRow.append('th')
                cell.html(header)
            })

            var tbodyRow = tbody.append('tr')
            values.map(value => {
                var cell = tbodyRow.append('td')
                cell.html(value)
            })
        }
    })
}
