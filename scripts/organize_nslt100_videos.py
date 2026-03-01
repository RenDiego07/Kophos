"""
Script para organizar videos del subset NSLT-100.

Este script:
1. Lee nslt_100.json para obtener los video_ids necesarios
2. Verifica que los videos NO estén en missing.txt
3. Copia los videos disponibles de data/raw/videos/ a data/raw/videos_nslt_100/
4. Organiza los videos en subcarpetas train/test/val según el subset
"""

import json
import os
import shutil
from pathlib import Path
from typing import Set, List, Dict
from collections import defaultdict


def load_nslt_100(filepath: str) -> Dict[str, str]:
    """Carga el archivo nslt_100.json y retorna dict de {video_id: subset}."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Extraer video_id -> subset
    return {video_id: info['subset'] for video_id, info in data.items()}


def load_missing_videos(filepath: str) -> Set[str]:
    """Carga el archivo missing.txt y retorna set de video_ids faltantes."""
    if not os.path.exists(filepath):
        print(f"⚠️  Archivo {filepath} no encontrado, continuando sin filtro...")
        return set()
    
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def get_available_videos(video_data: Dict[str, str], missing_ids: Set[str], 
                         videos_dir: Path) -> tuple:
    """
    Filtra video_ids que:
    1. NO están en missing.txt
    2. Existen físicamente en videos_dir
    
    Returns:
        tuple: (available_dict, not_available_list)
        available_dict: {subset: [video_ids]}
        not_available_list: [video_ids]
    """
    available = defaultdict(list)
    not_available = []
    
    for video_id, subset in video_data.items():
        # Saltar si está en missing.txt
        if video_id in missing_ids:
            continue
        
        # Verificar si el archivo existe
        video_file = videos_dir / f"{video_id}.mp4"
        if video_file.exists():
            available[subset].append(video_id)
        else:
            not_available.append(video_id)
    
    return dict(available), not_available


def copy_videos_by_subset(videos_by_subset: Dict[str, List[str]], 
                          source_dir: Path, dest_dir: Path) -> Dict[str, int]:
    """
    Copia los videos desde source_dir a dest_dir organizados en subcarpetas.
    
    Args:
        videos_by_subset: {subset: [video_ids]}
        source_dir: Directorio fuente
        dest_dir: Directorio destino base
    
    Returns:
        Dict con estadísticas por subset
    """
    stats = {}
    total_copied = 0
    total_errors = 0
    
    for subset, video_ids in videos_by_subset.items():
        # Crear subcarpeta para este subset
        subset_dir = dest_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Copiando videos de '{subset}' a: {subset_dir}")
        print(f"   Total videos: {len(video_ids)}")
        
        copied = 0
        errors = 0
        
        for video_id in video_ids:
            source_file = source_dir / f"{video_id}.mp4"
            dest_file = subset_dir / f"{video_id}.mp4"
            
            try:
                shutil.copy2(source_file, dest_file)
                copied += 1
                if copied % 50 == 0:
                    print(f"   Copiados: {copied}/{len(video_ids)}")
            except Exception as e:
                print(f"   ❌ Error copiando {video_id}: {e}")
                errors += 1
        
        stats[subset] = {'copied': copied, 'errors': errors}
        total_copied += copied
        total_errors += errors
        
        print(f"   ✅ {subset}: {copied} videos copiados")
        if errors > 0:
            print(f"   ❌ {subset}: {errors} errores")
    
    print(f"\n✅ Proceso completado!")
    print(f"   Total videos copiados: {total_copied}")
    if total_errors > 0:
        print(f"   Total errores: {total_errors}")
    
    return stats


def main():
    # Rutas base
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    
    nslt_100_file = raw_dir / "nslt_100.json"
    missing_file = raw_dir / "missing.txt"
    videos_dir = raw_dir / "videos"
    output_dir = raw_dir / "videos_nslt_100"
    
    print("=" * 60)
    print("🎬 Organizador de Videos NSLT-100")
    print("=" * 60)
    
    # 1. Cargar nslt_100.json
    print(f"\n📖 Cargando {nslt_100_file}...")
    nslt_video_data = load_nslt_100(nslt_100_file)
    print(f"   Videos en NSLT-100: {len(nslt_video_data)}")
    
    # 2. Cargar missing.txt
    print(f"\n📖 Cargando {missing_file}...")
    missing_ids = load_missing_videos(missing_file)
    print(f"   Videos en missing.txt: {len(missing_ids)}")
    
    # 3. Filtrar videos disponibles
    print(f"\n🔍 Verificando disponibilidad en {videos_dir}...")
    available_by_subset, not_available_ids = get_available_videos(
        nslt_video_data, missing_ids, videos_dir
    )
    
    # Calcular totales
    total_available = sum(len(videos) for videos in available_by_subset.values())
    
    print(f"\n📊 Resumen:")
    print(f"   Videos requeridos (nslt_100): {len(nslt_video_data)}")
    print(f"   Videos en missing.txt: {len(missing_ids)}")
    print(f"   Videos disponibles: {total_available}")
    for subset in sorted(available_by_subset.keys()):
        print(f"      - {subset}: {len(available_by_subset[subset])} videos")
    print(f"   Videos NO disponibles: {len(not_available_ids)}")
    
    if not_available_ids:
        print(f"\n⚠️  Videos NO disponibles (primeros 10):")
        for vid in list(not_available_ids)[:10]:
            print(f"      - {vid}")
        if len(not_available_ids) > 10:
            print(f"      ... y {len(not_available_ids) - 10} más")
    
    # 4. Copiar videos
    if total_available > 0:
        print(f"\n¿Deseas copiar {total_available} videos a {output_dir}? (y/n): ", end='')
        response = input().strip().lower()
        
        if response == 'y':
            stats = copy_videos_by_subset(available_by_subset, videos_dir, output_dir)
            
            # Crear archivo de resumen
            summary_file = output_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"NSLT-100 Videos Organizados\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Videos requeridos: {len(nslt_video_data)}\n")
                f.write(f"Videos en missing.txt: {len(missing_ids)}\n")
                f.write(f"Videos disponibles: {total_available}\n")
                f.write(f"Videos NO disponibles: {len(not_available_ids)}\n\n")
                
                f.write(f"Distribución por subset:\n")
                for subset in sorted(available_by_subset.keys()):
                    f.write(f"\n{subset.upper()}:\n")
                    f.write(f"  Total: {len(available_by_subset[subset])} videos\n")
                    f.write(f"  Copiados: {stats[subset]['copied']}\n")
                    if stats[subset]['errors'] > 0:
                        f.write(f"  Errores: {stats[subset]['errors']}\n")
                    f.write(f"  Videos:\n")
                    for vid in sorted(available_by_subset[subset]):
                        f.write(f"    {vid}.mp4\n")
                
                if not_available_ids:
                    f.write(f"\n\nVideos NO disponibles:\n")
                    for vid in sorted(not_available_ids):
                        f.write(f"  {vid}\n")
            
            print(f"\n📝 Resumen guardado en: {summary_file}")
        else:
            print("\n❌ Operación cancelada")
    else:
        print("\n❌ No hay videos disponibles para copiar")


if __name__ == "__main__":
    main()
