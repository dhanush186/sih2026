const audioFile = document.getElementById("audioFile");
const analyzeButton = document.getElementById("analyzeButton");
const result = document.getElementById("result");


analyzeButton.addEventListener("click", async function () {

    if (audioFile.files.length === 0) {

        result.innerHTML = `
            <h3>Please select an audio file</h3>

            <p>
                Select a voice recording before starting the analysis.
            </p>
        `;

        return;
    }


    const selectedFile = audioFile.files[0];


    result.innerHTML = `
        <h3>Preparing Analysis...</h3>

        <p>
            File: ${selectedFile.name}
        </p>
    `;


    const formData = new FormData();

    formData.append("audio", selectedFile);


    try {

        result.innerHTML = `
            <h3>Analyzing Voice...</h3>

            <p>
                Please wait while the recording is being analyzed.
            </p>
        `;


        /*
         * REAL BACKEND REQUEST WILL BE ADDED HERE
         *
         * Example:
         *
         * const response = await fetch(
         *     API_BASE_URL + "/analyze",
         *     {
         *         method: "POST",
         *         body: formData
         *     }
         * );
         *
         * const data = await response.json();
         */


        // Temporary mock result for frontend testing

        await new Promise(function (resolve) {

            setTimeout(resolve, 2000);

        });


        const mockResult = {

            fileName: selectedFile.name,

            aiProbability: 87,

            speakerMatch: 92,

            riskScore: 78,

            riskLevel: "HIGH"

        };


        displayResult(mockResult);


    } catch (error) {

        console.error(error);

        result.innerHTML = `
            <h3>Analysis Failed</h3>

            <p>
                Something went wrong while analyzing the voice.
            </p>

            <p>
                Please try again.
            </p>
        `;

    }

});


function displayResult(data) {

    result.innerHTML = `

        <h3>Analysis Result</h3>

        <p>
            <strong>File:</strong>
            ${data.fileName}
        </p>

        <p>
            <strong>AI Probability:</strong>
            ${data.aiProbability}%
        </p>

        <p>
            <strong>Speaker Match:</strong>
            ${data.speakerMatch}%
        </p>

        <p>
            <strong>Risk Score:</strong>
            ${data.riskScore}/100
        </p>

        <p>
            <strong>Risk Level:</strong>
            ${data.riskLevel}
        </p>

        <p>
            ⚠️ This is currently a mock result.
        </p>

    `;

}