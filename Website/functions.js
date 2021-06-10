function showPage(menuId, id){
    
    d3.selectAll('.activeMenuItem').attr('class', '');
    d3.selectAll('.page-div').style('display', 'none');

    const element = d3.select("#"+id); 
    element.style('display', 'block');

    d3.select("#"+menuId).attr("class", "activeMenuItem")
}



function showSubMenuPage(menuId, subMenuId, id) {

    showPage(menuId, id)

    d3.selectAll('.dropdown').style('display', 'none');
    if (subMenuId) {
        const element = d3.select("#"+subMenuId);
        element.style('display', 'block')
    }

    if (id === 'regression')
        showPlots('sub3a', "Linear_Regression")
    else if (id === "classification")
        showReport("sub4a", "Logistic_Classification")
}

function showPlots(menuId, plot) {
    d3.selectAll('.activeMenuItem').attr('class', '');
    d3.select("#"+menuId).attr("class", "activeMenuItem")
    plots(plot)
}

function showReport(menuId, rep) {
    d3.selectAll('.activeMenuItem').attr('class', '');
    d3.select("#"+menuId).attr("class", "activeMenuItem")
    report(rep)
}

currentSlide = 0
totalSlides = 11

function nextSlide() {
    d3.selectAll('.content').style('display', 'none');
    currentSlide = (currentSlide + 1) % totalSlides
    d3.select("#obs"+currentSlide).style("display", "block")
}

function prevSlide() {
    d3.selectAll('.content').style('display', 'none');
    currentSlide = (currentSlide - 1 + totalSlides) % totalSlides
    d3.select("#obs"+currentSlide).style("display", "block")
}
