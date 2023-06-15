namespace DeepSharp.RL.Environs
{
    /// <summary>
    ///     片段
    /// </summary>
    public class Episode
    {
        public Episode()
        {
            Steps = new List<Step>();
            SumReward = new Reward(0);
            DateTime = DateTime.Now;
        }

        public List<Step> Steps { set; get; }
        public Step this[int i] => Steps[i];

        public Reward SumReward { set; get; }
        public DateTime DateTime { set; get; }

        public bool IsComplete { set; get; }
        public int Length => Steps.Count;
    }
}