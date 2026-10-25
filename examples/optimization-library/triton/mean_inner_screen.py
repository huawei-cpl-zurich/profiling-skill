"""Conservative ELF/ABI screen; equal text alone never certifies a no-op."""
import hashlib
import struct


def elf_state(data,entry):
    if not isinstance(data,bytes) or data[:6]!=b'\x7fELF\x02\x01':raise ValueError('ELF64 little-endian required')
    header=struct.unpack_from('<HHIQQQIHHHHHH',data,16);offset=header[5];size,count,names=header[10:13]
    if size!=64 or offset+count*size>len(data):raise ValueError('invalid section table')
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
    table=sections[names];strings=data[table[4]:table[4]+table[5]]
    def name(t):return strings[t[0]:].split(b'\0',1)[0].decode()
    def raw(t):
        if t[1]==8:return b''
        if t[4]+t[5]>len(data):raise ValueError('section outside file')
        return data[t[4]:t[4]+t[5]]
    text=[(i,t) for i,t in enumerate(sections) if name(t)=='.text']
    if len(text)!=1 or not text[0][1][5]:raise ValueError('nonempty executable text required')
    text_index,text_section=text[0];alloc=[];relocs=[];known=True
    for index,t in enumerate(sections):
        content=raw(t)
        if t[2]&2 and name(t)!='.text':alloc.append(dict(name=name(t),type=t[1],flags=t[2],address=t[3],size=t[5],alignment=t[8],sha256=hashlib.sha256(content).hexdigest()))
        if t[1] not in (4,9):continue
        target=t[7];symtab=sections[t[6]];strtab=sections[symtab[6]];symbols=raw(symtab);symbol_names=raw(strtab)
        step=24 if t[1]==4 else 16
        if t[9]!=step or symtab[9]!=24:known=False;continue
        for pos in range(0,len(content),step):
            address,info=struct.unpack_from('<QQ',content,pos);sym_index=info>>32
            sym=struct.unpack_from('<IBBHQQ',symbols,sym_index*24)
            symbol=symbol_names[sym[0]:].split(b'\0',1)[0].decode();symbol='<entry>' if symbol==entry else symbol
            target_name=name(sections[target]) if target<len(sections) else str(target)
            relocs.append(dict(target_section=target_name,offset=address,kind=info&0xffffffff,
                               addend=struct.unpack_from('<q',content,pos+16)[0] if t[1]==4 else None,
                               symbol=dict(name=symbol,info=sym[1],other=sym[2],section=name(sections[sym[3]]) if sym[3]<len(sections) else sym[3],value=sym[4],size=sym[5])))
    # Nonalloc loader notes/metadata remain significant; no debug/string-table byte comparison substitutes for semantic relocation targets.
    loader=[dict(name=name(t),sha256=hashlib.sha256(raw(t)).hexdigest()) for t in sections if not t[2]&2 and ('note' in name(t) or 'metadata' in name(t) or 'ascend' in name(t))]
    return dict(full_sha256=hashlib.sha256(data).hexdigest(),elf_abi=dict(type=header[0],machine=header[1],version=header[2],entry=header[3],flags=header[6]),text_sha256=hashlib.sha256(raw(text_section)).hexdigest(),text_bytes=text_section[5],allocated_data=alloc,semantic_relocations=relocs,relocations_decoded=known,loader_metadata=loader)


def compare(before,after,abi_before,abi_after):
    if before['full_sha256']==after['full_sha256']:return 'compiler-noop-full-ELF'
    abi_known=abi_before.get('known') is True and abi_after.get('known') is True
    same_abi=abi_known and abi_before==abi_after and before['elf_abi']==after['elf_abi']
    decoded=before['relocations_decoded'] and after['relocations_decoded']
    same_state=all(before[k]==after[k] for k in ('allocated_data','semantic_relocations','loader_metadata'))
    if before['text_sha256']==after['text_sha256']:
        return 'compiler-noop-effective' if same_abi and decoded and same_state else 'inconclusive-identical-text'
    # Relocation-bearing differences need architecture-specific decoded instruction proof; don't manufacture a contrast.
    text_relocs=[x for state in (before,after) for x in state['semantic_relocations'] if x['target_section']=='.text']
    return 'code-distinct' if same_abi and decoded and not text_relocs and same_state else 'inconclusive-code-or-ABI'
