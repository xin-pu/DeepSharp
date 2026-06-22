// Stats panel update module
const StatsPanel = (() => {
    function update(progress) {
        setValue("stat-episode", progress.episodeCount);
        setValue("stat-steps", progress.stepCount);
        setValue("stat-sum-reward", progress.sumReward != null ? progress.sumReward.toFixed(3) : "—");
        setValue("stat-avg-reward", progress.averageReward != null ? progress.averageReward.toFixed(3) : "—");
        setValue("stat-epsilon", progress.epsilon != null ? progress.epsilon.toFixed(4) : "—");
        setValue("stat-loss", progress.loss != null ? progress.loss.toFixed(4) : "—");
    }

    function setValue(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    function reset() {
        setValue("stat-episode", "0");
        setValue("stat-steps", "0");
        setValue("stat-sum-reward", "0");
        setValue("stat-avg-reward", "0");
        setValue("stat-epsilon", "—");
        setValue("stat-loss", "—");
    }

    return { update, reset };
})();
