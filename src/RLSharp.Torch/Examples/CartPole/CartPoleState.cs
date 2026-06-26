namespace RLSharp.Torch.Examples.CartPole
{
	public sealed record CartPoleState(
		float Position,
		float Velocity,
		float Angle,
		float AngularVelocity)
	{
		public float[] ToFeatures()
		{
			return [Position, Velocity, Angle, AngularVelocity];
		}
	}
}