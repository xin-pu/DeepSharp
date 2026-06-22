// SignalR client wrapper for FrozenLake training hub
const SignalRClient = (() => {
    const connection = new signalR.HubConnectionBuilder()
        .withUrl("/trainingHub")
        .withAutomaticReconnect()
        .configureLogging(signalR.LogLevel.Warning)
        .build();

    async function start() {
        if (connection.state === signalR.HubConnectionState.Disconnected) {
            try {
                await connection.start();
                console.log("SignalR connected");
            } catch (err) {
                console.error("SignalR connection error:", err);
                // Retry after 3s
                setTimeout(start, 3000);
            }
        }
    }

    function onStepUpdate(callback) {
        connection.on("StepUpdate", callback);
    }

    function onEpisodeEnd(callback) {
        connection.on("EpisodeEnd", callback);
    }

    function onTrainingStarted(callback) {
        connection.on("TrainingStarted", callback);
    }

    function onTrainingStopped(callback) {
        connection.on("TrainingStopped", callback);
    }

    function onError(callback) {
        connection.on("Error", callback);
    }

    async function startTraining(config) {
        await connection.invoke("StartTraining", config);
    }

    async function stopTraining() {
        await connection.invoke("StopTraining");
    }

    async function runDemo() {
        await connection.invoke("RunDemo");
    }

    async function resetEnv() {
        await connection.invoke("ResetEnv");
    }

    return {
        start,
        onStepUpdate,
        onEpisodeEnd,
        onTrainingStarted,
        onTrainingStopped,
        onError,
        startTraining,
        stopTraining,
        runDemo,
        resetEnv
    };
})();
