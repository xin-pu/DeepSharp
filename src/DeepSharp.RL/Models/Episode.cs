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
            DateTime = DateTime.Now;
        }

        public List<Step> Oars { set; get; }
        public Reward SumReward { set; get; }
        public DateTime DateTime { set; get; }

        public int Steps => Oars.Count;
    }
}