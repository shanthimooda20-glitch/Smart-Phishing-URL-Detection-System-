/**
 * Scanner interactions: example chips, client-side guard rails and the
 * loading state shown while the server analyses the URL.
 */
(function () {
    "use strict";

    var form = document.getElementById("scan-form");
    if (!form) { return; }

    var input = document.getElementById("url-input");
    var button = document.getElementById("scan-button");
    var progress = document.getElementById("scan-progress");
    var status = document.getElementById("scan-status");

    var STAGES = [
        "Validating URL structure…",
        "Extracting lexical features…",
        "Scoring against the trained model…",
        "Building the explanation…"
    ];

    Array.prototype.forEach.call(document.querySelectorAll(".js-example"), function (chip) {
        chip.addEventListener("click", function () {
            input.value = chip.textContent.trim();
            input.focus();
        });
    });

    /** Show the progress bar and cycle through the pipeline stages. */
    function startLoading() {
        button.disabled = true;
        progress.classList.remove("hidden");
        status.classList.remove("hidden");

        var index = 0;
        status.textContent = STAGES[0];
        var timer = window.setInterval(function () {
            index += 1;
            if (index >= STAGES.length) {
                window.clearInterval(timer);
                return;
            }
            status.textContent = STAGES[index];
        }, 450);
    }

    form.addEventListener("submit", function (event) {
        var value = (input.value || "").trim();
        if (!value) {
            event.preventDefault();
            input.focus();
            return;
        }
        // The server re-validates everything; this only avoids a pointless
        // round trip for obviously unsupported input.
        if (/^\s*(javascript|data|file|vbscript):/i.test(value)) {
            event.preventDefault();
            status.classList.remove("hidden");
            status.textContent = "Only http:// and https:// URLs can be analysed.";
            return;
        }
        startLoading();
    });
}());
