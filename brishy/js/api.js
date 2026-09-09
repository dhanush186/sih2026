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

    const fileName = audioFile.files[0].name;

    result.innerHTML = `
        <h3>Selected Voice</h3>

        <p>
            <strong>File:</strong> ${fileName}
        </p>

        <p>
            Ready for analysis.
        </p>
    `;

});