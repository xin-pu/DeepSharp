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
        if (!gridState || !gridState.cells) return;

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

    return { init, render };
})();
