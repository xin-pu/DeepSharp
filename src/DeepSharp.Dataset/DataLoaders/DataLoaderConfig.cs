namespace DeepSharp.Dataset
{
	public struct DataLoaderConfig
	{
		public DataLoaderConfig()
		{
			Seed = null;
		}

		public int BatchSize { get; set; } = 4;

		public bool Shuffle { get; set; } = true;

		public bool DropLast { get; set; } = true;

		public int NumWorker { get; set; } = 1;

		public int? Seed { get; set; }

		public torch.Device Device { get; set; } = new(DeviceType.CUDA);
	}
}