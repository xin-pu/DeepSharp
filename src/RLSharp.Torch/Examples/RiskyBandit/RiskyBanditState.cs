namespace RLSharp.Torch.Examples.RiskyBandit
{
	public sealed record RiskyBanditState(int Step, int LastAction, float TotalReward)
	{
		public float[] ToFeatures()
		{
			return [Step / 100f, LastAction / 2f, TotalReward / 100f];
		}
	}
}