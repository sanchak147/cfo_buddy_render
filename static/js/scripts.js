// scripts.js

// Function to dynamically handle the form submission
function handleSubmit() {
    // Get the query and other input values
    const query = document.getElementById("query").value;
    const companyDomain = document.getElementById("companyDomain").value;
    const productFocus = document.getElementById("productFocus").value;
    const employeeStrength = document.getElementById("employeeStrength").value;

    // Show a loading message while processing
    document.getElementById("response").innerHTML = "Processing your query, please wait...";

    // Prepare data to send in the request
    const data = {
        query: query,
        companyDomain: companyDomain,
        productFocus: productFocus,
        employeeStrength: employeeStrength
    };

    // Send data via POST to the Flask route
    fetch('/process_query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        // Render the response data
        document.getElementById("response").innerHTML = `
            <h3>Analysis:</h3>
            <p>${data.text1 || "No introduction provided."}</p>
            <h3>Comparative Table:</h3>
            <table border="1">
                <thead>
                    <tr>
                        <th>2021</th>
                        <th>2022</th>
                        <th>2023</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.table ? data.table.map(row => `
                        <tr>
                            <td>${row['2021']}</td>
                            <td>${row['2022']}</td>
                            <td>${row['2023']}</td>
                        </tr>
                    `).join('') : "<tr><td colspan='3'>No table data available</td></tr>"}
                </tbody>
            </table>
            <h3>Conclusion:</h3>
            <p>${data.text2 || "No conclusion provided."}</p>
            <h3>Visualization:</h3>
            <img src="${data.chart_url || ''}" alt="Generated Chart" style="max-width: 100%; height: auto;" />
        `;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("response").innerHTML = "Sorry, there was an error processing your query.";
    });
}
