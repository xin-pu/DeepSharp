// SignalR client wrapper for FrozenLake training hub
const SignalRClient = (() => {
    const connection = new signalR.HubConnectionBuilder()
        .withUrl("/trainingHub")
        .withAutomaticReconnect()
        .configureLogging(signalR.LogLevel.Information)
        .build();

    let onConnectedCallback = null;
    let onDisconnectedCallback = null;

    connection.onreconnected(() => {
        console.log("SignalR reconnected");
        if (onConnectedCallback) onConnectedCallback();
    });

    connection.onclose(() => {
        console.warn("SignalR connection closed");
        if (onDisconnectedCallback) onDisconnectedCallback();
    });

    async function start() {
        if (connection.state === signalR.HubConnectionState.Disconnected) {
            try {
                await connection.start();
                console.log("SignalR connected");
                if (onConnectedCallback) onConnectedCallback();
            } catch (err) {
                console.error("SignalR connection error:", err);
                if (onDisconnectedCallback) onDisconnectedCallback();
                // Retry after 3s
                setTimeout(start, 3000);
            }
        }
    }

    function isConnected() {
        return connection.state === signalR.HubConnectionState.Connected;
    }

    function onConnectionChange(connectedCb, disconnectedCb) {
        onConnectedCallback = connectedCb;
        onDisconnectedCallback = disconnectedCb;
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
        if (!isConnected()) {
            throw new Error("SignalR not connected. Please wait...");
        }
        await connection.invoke("StartTraining", config);
    }

    async function stopTraining() {
        await connection.invoke("StopTraining");
    }

    async function runDemo() {
        if (!isConnected()) {
            throw new Error("SignalR not connected. Please wait...");
        }
        await connection.invoke("RunDemo");
    }

    async function resetEnv() {
        if (!isConnected()) {
            throw new Error("SignalR not connected. Please wait...");
        }
        await connection.invoke("ResetEnv");
    }

    return {
        start,
        isConnected,
        onConnectionChange,
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
