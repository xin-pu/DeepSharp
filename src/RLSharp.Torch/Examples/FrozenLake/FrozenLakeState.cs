namespace RLSharp.Torch.Examples.FrozenLake
{
	public sealed record FrozenLakeState(int PlayerIndex, int Size)
	{
		public float[] ToOneHot()
		{
			var values = new float[Size * Size];
			values[PlayerIndex] = 1f;
			return values;
		}
	}
}