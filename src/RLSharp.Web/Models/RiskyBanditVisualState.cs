namespace RLSharp.Web.Models
{
	public sealed class RiskyBanditVisualState
	{
		public string EnvironmentType { get; set; } = "RiskyBandit";

		public int Step { get; set; }

		public int LastAction { get; set; }

		public string ActionName { get; set; } = "";

		public float Reward { get; set; }

		public float TotalReward { get; set; }

		public bool IsComplete { get; set; }
	}
}