namespace DeepSharp.Dataset
{
    public struct DataLoaderConfig
    {
        public DataLoaderConfig()
        {
            Seed = null;
        }

        public int BatchSize { set; get; } = 4;
        public bool Shuffle { set; get; } = true;
        public bool DropLast { set; get; } = true;
        public int NumWorker { set; get; } = 1;
        public int? Seed { set; get; }
        public torch.Device Device { set; get; } = new(DeviceType.CUDA);
    }
}