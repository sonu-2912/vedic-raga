"""Inspect completed local ZIP members while the official download continues."""
import io,struct,zlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
file=root/'data/private/gamer/data.zip.partial'
out=root/'reports/gamer_structure';out.mkdir(exist_ok=True)
with file.open('rb') as f:
    for i in range(500):
        header=f.read(30)
        if len(header)<30 or header[:4]!=b'PK\x03\x04':break
        p=struct.unpack('<4s5H3I2H',header)
        name=f.read(p[9]).decode(errors='replace');extra=f.read(p[10])
        data=bytearray();complete=True
        if p[3]==8:
            decoder=zlib.decompressobj(-15)
            while not decoder.eof:
                chunk=f.read(1024*1024)
                if not chunk:complete=False;break
                data.extend(decoder.decompress(chunk))
            if decoder.eof:f.seek(-len(decoder.unused_data),1)
        else:data.extend(f.read(p[7]))
        print(json.dumps({'member':name,'uncompressed_bytes_so_far':len(data),'complete':complete}),flush=True)
        if not complete:break
        if p[2]&8:
            d=f.read(4);f.read(12 if d==b'PK\x07\x08' else 8)
        if name.endswith('.txt'):
            (out/'first_header.txt').write_bytes(data[:5000]); print(data[:2000].decode(errors='replace'));break
        if name.endswith('.h5'):
            target=out/Path(name).name
            if not target.exists():target.write_bytes(data)
            print('Saved complete HDF5: '+str(target));break
