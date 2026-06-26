// Canvas-based 4x4 FrozenLake grid renderer
const GridRenderer = (() => {
    const CELL_SIZE = 100;
    const PADDING = 10;
    const GRID_SIZE = 4;
    const CANVAS_SIZE = CELL_SIZE * GRID_SIZE + PADDING * 2;

    let canvas, ctx;
    let lastGridState = null;
    let pendingGridState = null;
    let animFrameId = null;
    let environmentType = "FrozenLake";

    // Color mapping
    const ROLE_COLORS = {
        "Start": "#3fb950",
        "Ice": "#8dd5e6",
        "Hole": "#444",
        "End": "#e3b341"
    };

    const ROLE_LABELS = {
        "Start": "S",
        "Ice": "",
        "Hole": "H",
        "End": "G"
    };

    const ROLE_BORDER = {
        "Start": "#2ea043",
        "Ice": "#5a9eaa",
        "Hole": "#222",
        "End": "#c49820"
    };

    function init() {
        canvas = document.getElementById("grid-canvas");
        ctx = canvas.getContext("2d");
    }

    function render(gridState) {
        if (!canvas || !ctx) return;

        // Throttle via requestAnimationFrame
        pendingGridState = gridState;
        if (!animFrameId) {
            animFrameId = requestAnimationFrame(() => {
                animFrameId = null;
                doRender(pendingGridState || gridState);
                lastGridState = pendingGridState;
                pendingGridState = null;
            });
        }
    }

    function doRender(gridState) {
        if (!gridState) return;

        if (gridState.environmentType === "CartPole" || environmentType === "CartPole") {
            drawCartPole(gridState);
            return;
        }

        if (gridState.environmentType === "RiskyBandit" || environmentType === "RiskyBandit") {
            drawRiskyBandit(gridState);
            return;
        }

        if (!gridState.cells) return;

        ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

        // Draw background
        ctx.fillStyle = "#0d1117";
        ctx.fillRect(PADDING, PADDING, CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE);

        // Draw cells
        for (let i = 0; i < gridState.cells.length; i++) {
            const cell = gridState.cells[i];
            const x = PADDING + cell.column * CELL_SIZE;
            const y = PADDING + cell.row * CELL_SIZE;

            drawCell(x, y, cell);
        }

        // Draw grid lines
        ctx.strokeStyle = "#30363d";
        ctx.lineWidth = 2;
        for (let i = 0; i <= GRID_SIZE; i++) {
            const pos = PADDING + i * CELL_SIZE;
            ctx.beginPath();
            ctx.moveTo(pos, PADDING);
            ctx.lineTo(pos, PADDING + CELL_SIZE * GRID_SIZE);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(PADDING, pos);
            ctx.lineTo(PADDING + CELL_SIZE * GRID_SIZE, pos);
            ctx.stroke();
        }

        // Update action/reward display
        document.getElementById("action-text").textContent = gridState.actionName || "-";
        document.getElementById("reward-text").textContent = gridState.reward != null ? gridState.reward.toFixed(2) : "-";
    }

    function setEnvironment(nextEnvironmentType) {
        environmentType = nextEnvironmentType || "FrozenLake";
        if (!canvas || !ctx) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (environmentType === "CartPole") {
            drawCartPole({
                environmentType: "CartPole",
                position: 0,
                angle: 0,
                actionName: "-",
                reward: 0
            });
        } else if (environmentType === "RiskyBandit") {
            drawRiskyBandit({
                environmentType: "RiskyBandit",
                step: 0,
                lastAction: -1,
                actionName: "-",
                reward: 0,
                totalReward: 0
            });
        }
    }

    function drawCartPole(state) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const width = canvas.width;
        const height = canvas.height;
        const groundY = height * 0.72;
        const centerX = width / 2;
        const cartScale = width / 5.2;
        const cartX = centerX + clamp(state.position || 0, -2.4, 2.4) * cartScale;
        const cartY = groundY - 38;
        const cartW = 84;
        const cartH = 36;
        const poleLength = 140;
        const angle = state.angle || 0;
        const pivotX = cartX;
        const pivotY = cartY - 4;
        const poleEndX = pivotX + Math.sin(angle) * poleLength;
        const poleEndY = pivotY - Math.cos(angle) * poleLength;

        ctx.fillStyle = "#0d1117";
        ctx.fillRect(0, 0, width, height);

        ctx.strokeStyle = "#30363d";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(32, groundY);
        ctx.lineTo(width - 32, groundY);
        ctx.stroke();

        ctx.strokeStyle = "#58a6ff";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(32, groundY + 22);
        ctx.lineTo(width - 32, groundY + 22);
        ctx.stroke();

        ctx.fillStyle = "#3fb950";
        roundRect(ctx, cartX - cartW / 2, cartY, cartW, cartH, 8);
        ctx.fill();
        ctx.strokeStyle = "#2ea043";
        ctx.lineWidth = 2;
        ctx.stroke();

        drawWheel(cartX - 26, cartY + cartH + 8);
        drawWheel(cartX + 26, cartY + cartH + 8);

        ctx.strokeStyle = "#e3b341";
        ctx.lineWidth = 8;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(pivotX, pivotY);
        ctx.lineTo(poleEndX, poleEndY);
        ctx.stroke();
        ctx.lineCap = "butt";

        ctx.fillStyle = "#f0e68c";
        ctx.beginPath();
        ctx.arc(pivotX, pivotY, 8, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#c9d1d9";
        ctx.font = "14px 'Segoe UI', sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`Position: ${(state.position || 0).toFixed(3)}`, 20, 28);
        ctx.fillText(`Angle: ${(angle * 180 / Math.PI).toFixed(2)} deg`, 20, 50);
        ctx.fillText(`Velocity: ${(state.velocity || 0).toFixed(3)}`, 20, 72);

        document.getElementById("action-text").textContent = state.actionName || "-";
        document.getElementById("reward-text").textContent = state.reward != null ? state.reward.toFixed(2) : "-";
    }

    function drawRiskyBandit(state) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const width = canvas.width;
        const height = canvas.height;
        const arms = [
            { name: "Safe", subtitle: "steady", color: "#3fb950" },
            { name: "Neutral", subtitle: "swingy", color: "#58a6ff" },
            { name: "Risky", subtitle: "volatile", color: "#f85149" }
        ];
        const selected = Number.isInteger(state.lastAction) ? state.lastAction : -1;

        ctx.fillStyle = "#0d1117";
        ctx.fillRect(0, 0, width, height);

        ctx.fillStyle = "#c9d1d9";
        ctx.font = "bold 20px 'Segoe UI', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Risky Bandit", width / 2, 38);

        const cardW = 118;
        const cardH = 190;
        const gap = 16;
        const startX = (width - (cardW * arms.length + gap * (arms.length - 1))) / 2;
        const top = 86;

        arms.forEach((arm, index) => {
            const x = startX + index * (cardW + gap);
            const isSelected = selected === index;

            roundRect(ctx, x, top, cardW, cardH, 8);
            ctx.fillStyle = isSelected ? "#21262d" : "#161b22";
            ctx.fill();
            ctx.strokeStyle = isSelected ? arm.color : "#30363d";
            ctx.lineWidth = isSelected ? 4 : 2;
            ctx.stroke();

            ctx.fillStyle = arm.color;
            ctx.beginPath();
            ctx.arc(x + cardW / 2, top + 64, 30, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "#0d1117";
            ctx.font = "bold 22px 'Segoe UI', sans-serif";
            ctx.fillText(String(index + 1), x + cardW / 2, top + 72);

            ctx.fillStyle = "#c9d1d9";
            ctx.font = "bold 16px 'Segoe UI', sans-serif";
            ctx.fillText(arm.name, x + cardW / 2, top + 120);

            ctx.fillStyle = "#8b949e";
            ctx.font = "13px 'Segoe UI', sans-serif";
            ctx.fillText(arm.subtitle, x + cardW / 2, top + 144);
        });

        ctx.fillStyle = "#c9d1d9";
        ctx.font = "15px 'Segoe UI', sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`Step: ${state.step || 0}`, 32, height - 58);
        ctx.fillText(`Total reward: ${(state.totalReward || 0).toFixed(2)}`, 32, height - 34);

        document.getElementById("action-text").textContent = state.actionName || "-";
        document.getElementById("reward-text").textContent = state.reward != null ? state.reward.toFixed(2) : "-";
    }

    function drawWheel(x, y) {
        ctx.fillStyle = "#8b949e";
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#30363d";
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function drawCell(x, y, cell) {
        const innerMargin = 4;
        const ix = x + innerMargin;
        const iy = y + innerMargin;
        const iw = CELL_SIZE - innerMargin * 2;
        const ih = CELL_SIZE - innerMargin * 2;
        const radius = 8;

        // Rounded rect
        roundRect(ctx, ix, iy, iw, ih, radius);

        const bgColor = ROLE_COLORS[cell.role] || "#8dd5e6";
        const borderColor = ROLE_BORDER[cell.role] || "#5a9eaa";

        ctx.fillStyle = bgColor;
        ctx.fill();
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Role label (top-left)
        const label = ROLE_LABELS[cell.role] || "";
        if (label) {
            ctx.fillStyle = "#fff";
            ctx.font = "bold 18px 'Segoe UI', sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillText(label, ix + 8, iy + 6);
        }

        // Player icon
        if (cell.isPlayer) {
            const cx = x + CELL_SIZE / 2;
            const cy = y + CELL_SIZE / 2;
            const pr = 18;

            // Glow
            ctx.beginPath();
            ctx.arc(cx, cy, pr + 6, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(255, 255, 100, 0.3)";
            ctx.fill();

            // Player circle
            ctx.beginPath();
            ctx.arc(cx, cy, pr, 0, Math.PI * 2);
            ctx.fillStyle = "#f0e68c";
            ctx.fill();
            ctx.strokeStyle = "#e3b341";
            ctx.lineWidth = 3;
            ctx.stroke();

            // "P" label
            ctx.fillStyle = "#222";
            ctx.font = "bold 16px 'Segoe UI', sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("P", cx, cy);
        }
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
    }

    return { init, render, setEnvironment };
})();
