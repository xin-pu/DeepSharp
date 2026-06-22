using System.Text;

namespace DeepSharp.RL.Environs
{
	/// <summary>
	///     A sequence of steps from an environment interaction.
	/// </summary>
	public class Episode
	{
		public Episode()
		{
			Steps     = new List<Step>();
			SumReward = new Reward(0);
			DateTime  = DateTime.Now;
			Evaluate  = 0;
		}

		public Episode(List<Step> steps)
		{
			Steps     = steps;
			SumReward = new Reward(0);
			DateTime  = DateTime.Now;
			Evaluate  = 0;
		}

		public List<Step> Steps { get; set; }

		public Step this[int i] => Steps[i];

		public Reward SumReward { get; set; }

		public DateTime DateTime { get; set; }

		public bool IsComplete { get; set; }

		public int Length => Steps.Count;

		public float Evaluate { get; set; }

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
				var state  = s.PreState.Value!.ToString(torch.numpy);
				var action = s.Action.Value!.ToInt32();
				var reward = s.Reward.Value;
				var line   = $"{state},{action},{reward}";
				str.AppendLine(line);
			});
			return str.ToString();
		}


		/// <summary>
		///     Get an episode where each step's reward is discounted by Gamma to estimate Q-value.
		/// </summary>
		/// <param name="gamma">Discount factor.</param>
		/// <returns>Episode with discounted rewards.</returns>
		public Episode GetReturnEpisode(float gamma = 0.9f)
		{
			var stepsWithReturn = new List<Step>();


			var sumR  = 0f;
			var steps = Steps;
			steps.Reverse();
			foreach (var s in steps)
			{
				sumR *= gamma;
				sumR += s.Reward.Value;

				var sNew = (Step)s.Clone();
				sNew.Reward = new Reward(sumR);
				stepsWithReturn.Add(sNew);
			}

			steps.Reverse();
			stepsWithReturn.Reverse();
			var res = new Episode(stepsWithReturn);
			return res;
		}
	}
}