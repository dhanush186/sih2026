const audioFile = document.getElementById("audioFile");
const analyzeButton = document.getElementById("analyzeButton");
const result = document.getElementById("result");

analyzeButton.addEventListener("click", function () {

    if (audioFile.files.length === 0) {

        result.innerHTML = `
            <p>Please select an audio file first.</p>
        `;

        return;
    }

    result.innerHTML = `
        <p>Analyzing voice...</p>
    `;

    setTimeout(function () {

        const fileName = audioFile.files[0].name;

        result.innerHTML = `

            <h3>Analysis Result</h3>

            <p>
                <strong>File:</strong>
                ${fileName}
            </p>

            <p>
                <strong>AI Probability:</strong>
                87%
            </p>

            <p>
                <strong>Speaker Match:</strong>
                92%
            </p>

            <p>
                <strong>Risk Score:</strong>
                78/100
            </p>

            <p>
                <strong>Risk Level:</strong>
                HIGH
            </p>

        `;

    }, 2000);

});