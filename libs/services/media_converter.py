# @title ffmpegでm4aに変換するコンバータークラス
import os
import ffmpeg

class FFmpegM4AConverter:
    """
    FFmpegを使用してさまざまな形式のオーディオファイルをM4Aファイルに変換するクラス。

    使用方法:
    1. インスタンスを作成します。必要に応じて、オーディオ設定を指定できます。
       シンプルな使い方
       converter = FFmpegM4AConverter()

       詳細な使い方
       converter = FFmpegM4AConverter(sample_rate=48000, bitrate=256000, channels=2, bits_per_sample=16)

    2. convertメソッドまたは__call__メソッドを使用して、ファイルを変換します。
       シンプルな使い方
       converter("input.wav", "output_directory")  # デフォルトでnormalizeが適用されます
       converter.convert("input.wav", "output_directory")  # デフォルトでnormalizeが適用されます

       詳細な使い方
       converter("input.mp3", "output_directory", normalize=False, vbr=True, metadata={"artist": "John Doe", "title": "Example"})

    対応する入力ファイル形式:
    - オーディオ形式: .aac, .ac3, .aif, .aiff, .alac, .amr, .ape, .flac, .m4a, .mp3, .ogg, .opus, .wav, ...
    - ビデオ形式: .avi, .flv, .mkv, .mov, .mp4, .mpeg, .webm, .wmv, ...

    変換されたM4Aファイルは、指定された出力ディレクトリに保存されます。
    """

    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_BITRATE = 192000
    DEFAULT_CHANNELS = 1
    DEFAULT_BITS_PER_SAMPLE = 16
    DEFAULT_ADJUST_VOLUME = True
    DEFAULT_TARGET_VOLUME = -10

    def __init__(self, sample_rate=None, bitrate=None, channels=None, bits_per_sample=None, adjust_volume=None, target_volume=None):
        self.sample_rate = sample_rate or self.DEFAULT_SAMPLE_RATE
        self.bitrate = bitrate or self.DEFAULT_BITRATE
        self.channels = channels or self.DEFAULT_CHANNELS
        self.bits_per_sample = bits_per_sample or self.DEFAULT_BITS_PER_SAMPLE
        self.adjust_volume = adjust_volume if adjust_volume is not None else self.DEFAULT_ADJUST_VOLUME
        self.target_volume = target_volume or self.DEFAULT_TARGET_VOLUME
        self.supported_extensions = self._get_supported_extensions()

    def _get_supported_extensions(self):
        return [
            '.3g2', '.3gp', '.aac', '.ac3', '.aif', '.aiff', '.alac', '.amr', '.ape',
            '.asf', '.au', '.avi', '.caf', '.dts', '.dtshd', '.dv', '.eac3', '.flac',
            '.flv', '.m2a', '.m2ts', '.m4a', '.m4b', '.m4p', '.m4r', '.m4v', '.mka',
            '.mkv', '.mod', '.mov', '.mp1', '.mp2', '.mp3', '.mp4', '.mpa', '.mpc',
            '.mpeg', '.mpg', '.mts', '.nut', '.oga', '.ogg', '.ogm', '.ogv', '.ogx',
            '.opus', '.ra', '.ram', '.rm', '.rmvb', '.shn', '.spx', '.tak', '.tga',
            '.tta', '.vob', '.voc', '.wav', '.weba', '.webm', '.wma', '.wmv', '.wv',
            '.y4m', '.aac', '.aif', '.aiff', '.aiffc', '.flac', '.iff', '.m4a', '.m4b',
            '.m4p', '.mid', '.midi', '.mka', '.mp3', '.mpa', '.oga', '.ogg', '.opus',
            '.pls', '.ra', '.ram', '.spx', '.tta', '.voc', '.vqf', '.w64', '.wav',
            '.wma', '.xm', '.3gp', '.a64', '.ac3', '.amr', '.drc', '.dv', '.flv',
            '.gif', '.h261', '.h263', '.h264', '.hevc', '.m1v', '.m4v', '.mkv', '.mov',
            '.mp2', '.mp4', '.mpeg', '.mpeg1video', '.mpeg2video', '.mpeg4', '.mpg',
            '.mts', '.mxf', '.nsv', '.nuv', '.ogg', '.ogv', '.ps', '.rec', '.rm',
            '.rmvb', '.roq', '.svi', '.ts', '.vob', '.webm', '.wmv', '.y4m', '.yuv'
        ]

    def _apply_filters(self, stream, normalize=False, equalizer=None):
        if normalize:
            stream = ffmpeg.filter(stream, 'dynaudnorm')
        if equalizer:
            stream = ffmpeg.filter(stream, 'equalizer', equalizer)
        return stream

    def _analyze_volume(self, input_file):
        try:
            stats = ffmpeg.probe(input_file)
            audio_stats = next((s for s in stats['streams'] if s['codec_type'] == 'audio'), None)
            if audio_stats:
                volume_mean = float(audio_stats['tags']['volume_mean'])
                volume_max = float(audio_stats['tags']['volume_max'])
                return volume_mean, volume_max
            else:
                print("No audio stream found in the input file.")
        except ffmpeg.Error as e:
            print(f"Error occurred during volume analysis: {e.stderr}")
        return None, None

    def _adjust_volume(self, stream, volume_mean, volume_max, target_volume):
        if volume_mean is not None and volume_max is not None:
            volume_adjustment = target_volume - volume_max
            stream = ffmpeg.filter(stream, 'volume', volume=f'{volume_adjustment}dB')
        return stream

    def _convert(self, input_file, output_path, normalize=False, equalizer=None, vbr=False, metadata=None):
        stream = ffmpeg.input(input_file)

        if normalize:
            stream = self._apply_filters(stream, normalize=True)
        else:
            if self.adjust_volume:
                volume_mean, volume_max = self._analyze_volume(input_file)
                if volume_mean is not None and volume_max is not None:
                    stream = self._adjust_volume(stream, volume_mean, volume_max, self.target_volume)

        stream = self._apply_filters(stream, equalizer=equalizer)

        kwargs = {
            'acodec': 'aac',
            'ar': self.sample_rate,
            'ac': self.channels,
        }
        if vbr:
            kwargs['vbr'] = 5
        else:
            kwargs['b:a'] = self.bitrate

        output_stream = ffmpeg.output(stream, output_path, **kwargs)

        try:
            # '-y' オプションを追加して、出力ファイルの自動上書きを許可
            ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            print("Conversion completed successfully.")
        except ffmpeg.Error as e:
            stdout = e.stdout.decode('utf-8') if e.stdout else "No stdout"
            stderr = e.stderr.decode('utf-8') if e.stderr else "No stderr"
            print(f"Error occurred during conversion: {stderr}")
            print(f"FFmpeg stdout: {stdout}")

    def convert(self, input_file, output_dir, normalize=True, equalizer=None, vbr=False, metadata=None):
        _, extension = os.path.splitext(input_file)
        if extension.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")

        output_file = os.path.splitext(os.path.basename(input_file))[0] + "_converted.m4a"
        output_path = os.path.join(output_dir, output_file)
        os.makedirs(output_dir, exist_ok=True)

        self._convert(input_file, output_path, normalize, equalizer, vbr, metadata)
        return output_path

    def __call__(self, input_file, output_dir, normalize=True, equalizer=None, vbr=False, metadata=None):
        return self.convert(input_file, output_dir, normalize, equalizer, vbr, metadata)

# Usage
# converter = FFmpegM4AConverter()
# m4a_path = converter(input_file, output_dir)

# If error check your ffmpeg condition
# !ffmpeg -version
# !ffprobe -version