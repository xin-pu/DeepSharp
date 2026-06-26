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

    const ENVIRONMENT_AGENTS = {
        "FrozenLake": ["QLearning", "SARSA", "MonteCarloOnPolicy", "MonteCarloOffPolicy", "DQN", "REINFORCE", "A2C", "PPO"],
        "CartPole": ["DQN", "PPO"],
        "RiskyBandit": ["QLearning"]
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
        const environmentSelect = document.getElementById("environment-select");
        const speedSlider = document.getElementById("speed-slider");
        const btnTrain = document.getElementById("btn-train");
        const btnStop = document.getElementById("btn-stop");
        const btnDemo = document.getElementById("btn-demo");
        const btnReset = document.getElementById("btn-reset");
        const manualButtons = document.querySelectorAll(".manual-action");

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

        environmentSelect.addEventListener("change", () => updateAgentOptions(environmentSelect.value));
        agentSelect.addEventListener("change", () => updateParams(agentSelect.value));
        updateAgentOptions(environmentSelect.value);
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

        manualButtons.forEach(button => {
            button.addEventListener("click", () => manualStep(button.dataset.action));
        });

        document.addEventListener("keydown", event => {
            const tag = event.target && event.target.tagName;
            if (tag === "INPUT" || tag === "SELECT" || tag === "BUTTON") return;

            const environmentType = environmentSelect.value;
            const key = event.key.toLowerCase();
            let action = null;

            if (environmentType === "FrozenLake") {
                action = {
                    arrowup: "Up",
                    arrowdown: "Down",
                    arrowleft: "Left",
                    arrowright: "Right"
                }[key];
            } else if (environmentType === "CartPole") {
                action = key === "a" || key === "arrowleft"
                    ? "Left"
                    : key === "d" || key === "arrowright"
                        ? "Right"
                        : null;
            } else if (environmentType === "RiskyBandit") {
                action = key === "1" ? "Safe" : key === "2" ? "Neutral" : key === "3" ? "Risky" : null;
            }

            if (action) {
                event.preventDefault();
                manualStep(action);
            }
        });
    }

    async function manualStep(action) {
        if (isTraining || !SignalRClient.isConnected()) return;
        const environmentType = document.getElementById("environment-select").value;
        try {
            await SignalRClient.manualStep(environmentType, action);
        } catch (err) {
            setStatus("Error: " + err.message, "error");
        }
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
            EnvironmentType: document.getElementById("environment-select").value,
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
        document.getElementById("environment-select").disabled = training;
        document.querySelectorAll(".manual-action").forEach(button => {
            button.disabled = training;
        });
    }

    function updateAgentOptions(environmentType) {
        const agentSelect = document.getElementById("agent-select");
        const allowedAgents = ENVIRONMENT_AGENTS[environmentType] || ENVIRONMENT_AGENTS.FrozenLake;
        const current = agentSelect.value;

        Array.from(agentSelect.options).forEach(option => {
            option.hidden = !allowedAgents.includes(option.value);
        });

        if (!allowedAgents.includes(current)) {
            agentSelect.value = allowedAgents[0];
        }

        updateParams(agentSelect.value);
        updateManualControls(environmentType);
        if (window.GridRenderer) {
            GridRenderer.setEnvironment(environmentType);
        }
    }

    function updateManualControls(environmentType) {
        document.querySelectorAll(".manual-action").forEach(button => {
            button.hidden = button.dataset.env !== environmentType;
        });
    }

    return { init, setTrainingState };
})();
