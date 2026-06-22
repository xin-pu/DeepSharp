// Main entry point - wires everything together
(function () {
    async function init() {
        GridRenderer.init();
        Controls.init();

        // Register SignalR event handlers
        SignalRClient.onStepUpdate((gridState) => {
            GridRenderer.render(gridState);
        });

        SignalRClient.onEpisodeEnd((progress) => {
            StatsPanel.update(progress);
        });

        SignalRClient.onTrainingStarted((agentType) => {
            Controls.setTrainingState(true);
            StatsPanel.reset();
            setStatus(`Training with ${agentType}...`, "training");
        });

        SignalRClient.onTrainingStopped(() => {
            Controls.setTrainingState(false);
            setStatus("Ready", "");
        });

        SignalRClient.onError((msg) => {
            setStatus(`Error: ${msg}`, "error");
        });

        // Start connection
        await SignalRClient.start();
    }

    function setStatus(msg, className) {
        const bar = document.getElementById("status-bar");
        bar.textContent = msg;
        bar.className = "status " + (className || "");
    }

    // Start when DOM ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
