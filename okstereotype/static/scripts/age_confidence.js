 var age_conf_data =  [
     {
       values: [
           { x : 0, y : .3066},
           { x : 1, y : .5660},
           { x : 2, y : .7570},
           { x : 3, y : .8816},
           { x : 4, y : .9473},
           { x : 5, y : .9786},
           { x : 6, y : .9928},
           { x : 7, y : .9976},
           { x : 8, y : .9993},
           { x : 9, y : .9997},
           { x : 10, y : .9999},
           { x : 11, y : 1.000},
           ],
           key: "Percent Essays Predicted Correctly",
           color: '#bd362f'
     }
   ];
nv.addGraph(function() {
   var chart = nv.models.lineChart();

   chart.xAxis
       .axisLabel('Within Years')
       .tickFormat(d3.format('.1d'));

   chart.yAxis
       .axisLabel('Percent Right').margin({left:85})
       .tickFormat(d3.format('.1%'));
   chart.margin({bottom: 60, left: 100, right:60});
   d3.select('#age-confidence-chart svg').datum(age_conf_data).transition().duration(500).call(chart);
   nv.utils.windowResize(chart.update);
   return chart;
 });
