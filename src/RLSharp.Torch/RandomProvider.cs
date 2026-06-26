namespace RLSharp.Torch
{
	/// <summary>
	///     Central random source used by managed-code exploration decisions.
	/// </summary>
	public static class RandomProvider
	{
		private static readonly object Sync    = new();
		private static          Random _random = new();

		public static void SetSeed(long seed)
		{
			lock (Sync)
			{
				_random = new Random(unchecked((int)seed));
			}

			torch.random.manual_seed(seed);
		}

		public static double NextDouble()
		{
			lock (Sync)
			{
				return _random.NextDouble();
			}
		}
	}
}