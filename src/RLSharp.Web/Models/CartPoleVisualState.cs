namespace RLSharp.Web.Models
{
	public sealed class CartPoleVisualState
	{
		public string EnvironmentType { get; set; } = "CartPole";

		public float Position { get; set; }

		public float Velocity { get; set; }

		public float Angle { get; set; }

		public float AngularVelocity { get; set; }

		public string ActionName { get; set; } = "";

		public float Reward { get; set; }

		public bool IsComplete { get; set; }
	}
}