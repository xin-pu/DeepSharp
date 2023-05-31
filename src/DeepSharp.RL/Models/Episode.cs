namespace DeepSharp.RL.Models
{
    /// <summary>
    ///     片段
    /// </summary>
    public class Episode
    {
        public Episode()
        {
            Oars = new List<Step>();
            SumReward = new Reward(0);
        }

        public List<Step> Oars { set; get; }
        public Reward SumReward { set; get; }
    }
}