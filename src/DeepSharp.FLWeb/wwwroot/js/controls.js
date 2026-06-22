// Controls panel binding and hyperparameter management
const Controls = (() => {
    // Parameter definitions per agent type
    const AGENT_PARAMS = {
        "QLearning": [
            { id: "epsilon", label: "Epsilon (ε)", value: 0.2, type: "float" },
            { id: "alpha", label: "Alpha (α)", value: 0.2, type: "float" },
            { id: "gamma", label: "Gamma (γ)", value: 0.9, type: "float" }
        ],
        "SARSA": [
            { id: "epsilon", label: "Epsilon (ε)", value: 0.2, type: "float" },
            { id: "alpha", label: "Alpha (α)", value: 0.2, type: "float" },
            { id: "gamma", label: "Gamma (γ)", value: 0.9, type: "float" }
        ],
        "MonteCarloOnPolicy": [
            { id: "epsilon", label: "Epsilon (ε)", value: 0.1, type: "float" },
            { id: "t", label: "T (max steps)", value: 50, type: "int" }
        ],
        "MonteCarloOffPolicy": [
            { id: "epsilon", label: "Epsilon (ε)", value: 0.1, type: "float" },
            { id: "t", label: "T (max steps)", value: 50, type: "int" }
        ],
        "DQN": [
            { id: "epsilon", label: "Epsilon (ε)", value: 0.1, type: "float" },
            { id: "gamma", label: "Gamma (γ)", value: 0.99, type: "float" },
            { id: "n", label: "N (target sync)", value: 1000, type: "int" },
            { id: "c", label: "C (replay cap)", value: 10000, type: "int" },
            { id: "batchSize", label: "Batch Size", value: 32, type: "int" }
        ],
        "REINFORCE": [
            { id: "gamma", label: "Gamma (γ)", value: 0.99, type: "float" },
            { id: "alpha", label: "Alpha (α)", value: 0.01, type: "float" },
            { id: "batchSize", label: "Batch Size", value: 4, type: "int" }
        ],
        "A2C": [
            { id: "gamma", label: "Gamma (γ)", value: 0.99, type: "float" },
            { id: "alpha", label: "Alpha (α)", value: 0.01, type: "float" },
            { id: "beta", label: "Beta (β)", value: 0.01, type: "float" },
            { id: "batchSize", label: "Batch Size", value: 4, type: "int" }
        ]
    };

    let isTraining = false;

    function init() {
        const agentSelect = document.getElementById("agent-select");
        const speedSlider = document.getElementById("speed-slider");
        const btnTrain = document.getElementById("btn-train");
        const btnStop = document.getElementById("btn-stop");
        const btnDemo = document.getElementById("btn-demo");
        const btnReset = document.getElementById("btn-reset");

        // Agent change -> update params
        agentSelect.addEventListener("change", () => updateParams(agentSelect.value));
        updateParams(agentSelect.value);

        // Speed slider
        speedSlider.addEventListener("input", () => {
            document.getElementById("speed-value").textContent = speedSlider.value;
        });

        // Train button
        btnTrain.addEventListener("click", async () => {
            if (isTraining) return;
            const config = collectConfig();
            await SignalRClient.startTraining(config);
        });

        // Stop button
        btnStop.addEventListener("click", async () => {
            await SignalRClient.stopTraining();
        });

        // Demo button
        btnDemo.addEventListener("click", async () => {
            if (isTraining) return;
            await SignalRClient.runDemo();
        });

        // Reset button
        btnReset.addEventListener("click", async () => {
            await SignalRClient.resetEnv();
        });
    }

    function updateParams(agentType) {
        const container = document.getElementById("params-container");
        container.innerHTML = "";

        const params = AGENT_PARAMS[agentType] || [];
        const stepAttr = params.some(p => p.type === "float") ? "step=\"0.01\"" : "";

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
            if (param.type === "float") {
                input.step = "0.01";
            }

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

        // Read param values
        const paramsContainer = document.getElementById("params-container");
        const inputs = paramsContainer.querySelectorAll("input");
        inputs.forEach(input => {
            const id = input.id.replace("param-", "");
            // Capitalize first letter for PascalCase
            const key = id.charAt(0).toUpperCase() + id.slice(1);
            if (input.step === "0.01") {
                config[key] = parseFloat(input.value) || 0;
            } else {
                config[key] = parseInt(input.value) || 0;
            }
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
