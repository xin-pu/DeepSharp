using System.Text;

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
            Evaluate = 0;
        }

        public List<Step> Steps { set; get; }
        public Step this[int i] => Steps[i];

        public Reward SumReward { set; get; }
        public DateTime DateTime { set; get; }

        public bool IsComplete { set; get; }
        public int Length => Steps.Count;

        public float Evaluate { set; get; }

        public void Enqueue(Step step)
        {
            Steps.Add(step);
        }

        public int[] GetAction()
        {
            var actions = Steps
                .Select(a => a.Action.Value!.ToInt32())
                .ToArray();
            return actions;
        }

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"Test By Agent: Get Reward {SumReward}");
            Steps.ForEach(s =>
            {
                var state = s.PreState.Value!.ToString(torch.numpy);
                var action = s.Action.Value!.ToInt32();
                var reward = s.Reward.Value;
                var line = $"{state},{action},{reward}";
                str.AppendLine(line);
            });
            return str.ToString();
        }
    }
}