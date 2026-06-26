// Controls panel binding and hyperparameter management
const Controls = (() => {
    const AGENT_PARAMS = {
        "QLearning": [
            { id: "epsilon", label: "Epsilon", value: 0.2, type: "float" },
            { id: "alpha", label: "Alpha", value: 0.2, type: "float" },
            { id: "gamma", label: "Gamma", value: 0.9, type: "float" }
        ],
        "SARSA": [
            { id: "epsilon", label: "Epsilon", value: 0.2, type: "float" },
            { id: "alpha", label: "Alpha", value: 0.2, type: "float" },
            { id: "gamma", label: "Gamma", value: 0.9, type: "float" }
        ],
        "MonteCarloOnPolicy": [
            { id: "epsilon", label: "Epsilon", value: 0.1, type: "float" },
            { id: "t", label: "T (max steps)", value: 50, type: "int" }
        ],
        "MonteCarloOffPolicy": [
            { id: "epsilon", label: "Epsilon", value: 0.1, type: "float" },
            { id: "t", label: "T (max steps)", value: 50, type: "int" }
        ],
        "DQN": [
            { id: "epsilon", label: "Epsilon", value: 0.1, type: "float" },
            { id: "gamma", label: "Gamma", value: 0.99, type: "float" },
            { id: "n", label: "N (target sync)", value: 1000, type: "int" },
            { id: "c", label: "C (replay cap)", value: 10000, type: "int" },
            { id: "batchSize", label: "Batch Size", value: 32, type: "int" }
        ],
        "REINFORCE": [
            { id: "gamma", label: "Gamma", value: 0.99, type: "float" },
            { id: "alpha", label: "Alpha", value: 0.01, type: "float" },
            { id: "batchSize", label: "Batch Size", value: 4, type: "int" }
        ],
        "A2C": [
            { id: "gamma", label: "Gamma", value: 0.99, type: "float" },
            { id: "alpha", label: "Alpha", value: 0.01, type: "float" },
            { id: "beta", label: "Beta", value: 0.01, type: "float" },
            { id: "batchSize", label: "Batch Size", value: 4, type: "int" }
        ],
        "PPO": [
            { id: "gamma", label: "Gamma", value: 0.99, type: "float" },
            { id: "alpha", label: "Learning Rate", value: 0.001, type: "float" },
            { id: "batchSize", label: "Batch Size", value: 4, type: "int" }
        ]
    };

    let isTraining = false;

    function setStatus(msg, className) {
        const bar = document.getElementById("status-bar");
        if (bar) {
            bar.textContent = msg;
            bar.className = "status " + (className || "");
        }
    }

    function init() {
        const agentSelect = document.getElementById("agent-select");
        const speedSlider = document.getElementById("speed-slider");
        const btnTrain = document.getElementById("btn-train");
        const btnStop = document.getElementById("btn-stop");
        const btnDemo = document.getElementById("btn-demo");
        const btnReset = document.getElementById("btn-reset");

        btnTrain.disabled = true;
        btnDemo.disabled = true;
        btnStop.disabled = true;
        btnReset.disabled = true;
        setStatus("Connecting...", "");

        SignalRClient.onConnectionChange(
            () => {
                btnTrain.disabled = false;
                btnDemo.disabled = false;
                btnReset.disabled = false;
                setStatus("Connected", "training");
            },
            () => {
                btnTrain.disabled = true;
                btnDemo.disabled = true;
                btnStop.disabled = true;
                btnReset.disabled = true;
                setStatus("Disconnected - retrying...", "error");
            }
        );

        agentSelect.addEventListener("change", () => updateParams(agentSelect.value));
        updateParams(agentSelect.value);

        speedSlider.addEventListener("input", () => {
            document.getElementById("speed-value").textContent = speedSlider.value;
        });

        btnTrain.addEventListener("click", async () => {
            if (isTraining) return;
            if (!SignalRClient.isConnected()) {
                setStatus("Not connected. Please wait...", "error");
                return;
            }
            try {
                const config = collectConfig();
                await SignalRClient.startTraining(config);
            } catch (err) {
                setStatus("Error: " + err.message, "error");
            }
        });

        btnStop.addEventListener("click", async () => {
            try {
                await SignalRClient.stopTraining();
            } catch (err) {
                setStatus("Error: " + err.message, "error");
            }
        });

        btnDemo.addEventListener("click", async () => {
            if (isTraining) return;
            if (!SignalRClient.isConnected()) {
                setStatus("Not connected. Please wait...", "error");
                return;
            }
            try {
                await SignalRClient.runDemo();
            } catch (err) {
                setStatus("Error: " + err.message, "error");
            }
        });

        btnReset.addEventListener("click", async () => {
            if (!SignalRClient.isConnected()) {
                setStatus("Not connected. Please wait...", "error");
                return;
            }
            try {
                await SignalRClient.resetEnv();
            } catch (err) {
                setStatus("Error: " + err.message, "error");
            }
        });
    }

    function updateParams(agentType) {
        const container = document.getElementById("params-container");
        container.innerHTML = "";

        const params = AGENT_PARAMS[agentType] || [];
        params.forEach(param => {
            const div = document.createElement("div");
            div.className = "param-group";

            const label = document.createElement("label");
            label.textContent = param.label;
            label.htmlFor = `param-${param.id}`;

            const input = document.createElement("input");
            input.type = "number";
            input.id = `param-${param.id}`;
            input.value = param.value;
            if (param.type === "float") input.step = "0.01";

            div.appendChild(label);
            div.appendChild(input);
            container.appendChild(div);
        });
    }

    function collectConfig() {
        const config = {
            AgentType: document.getElementById("agent-select").value,
            SpeedDelayMs: parseInt(document.getElementById("speed-slider").value),
            MaxEpisodes: parseInt(document.getElementById("max-episodes").value),
            SmoothTarget: 0.8,
            SmoothLeft: 0.1,
            SmoothRight: 0.1,
            Epsilon: 0.2,
            Gamma: 0.9,
            Alpha: 0.2,
            T: 50,
            N: 1000,
            C: 10000,
            BatchSize: 32,
            Beta: 0.01
        };

        const inputs = document.getElementById("params-container").querySelectorAll("input");
        inputs.forEach(input => {
            const id = input.id.replace("param-", "");
            const key = id.charAt(0).toUpperCase() + id.slice(1);
            config[key] = input.step === "0.01"
                ? (parseFloat(input.value) || 0)
                : (parseInt(input.value) || 0);
        });

        return config;
    }

    function setTrainingState(training) {
        isTraining = training;
        document.getElementById("btn-train").disabled = training;
        document.getElementById("btn-demo").disabled = training;
        document.getElementById("btn-stop").disabled = !training;
        document.getElementById("agent-select").disabled = training;
    }

    return { init, setTrainingState };
})();
