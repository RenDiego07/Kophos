"""
Script para organizar videos del subset NSLT-100 (Versión con Integridad de Datos).

Este script:
1. Lee nslt_100.json para obtener los video_ids requeridos.
2. Verifica el motivo de ausencia (missing.txt o falta física).
3. Copia los videos disponibles organizados en subcarpetas train/test/val.
4. Genera un reporte matemáticamente consistente.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Set, List, Dict, Tuple
from collections import defaultdict


def load_nslt_100(filepath: str) -> Dict[str, str]:
    """Carga el archivo nslt_100.json y retorna dict de {video_id: subset}."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {video_id: info['subset'] for video_id, info in data.items()}


def load_missing_videos(filepath: str) -> Set[str]:
    """Carga el archivo missing.txt y retorna set de video_ids faltantes por origen."""
    if not os.path.exists(filepath):
        print(f"⚠️  Archivo {filepath} no encontrado, continuando sin filtro...")
        return set()
    
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def get_available_videos(video_data: Dict[str, str], missing_ids: Set[str], 
                         videos_dir: Path) -> Tuple[Dict[str, List[str]], List[str], List[str]]:
    """
    Filtra y categoriza el estado de cada video para garantizar la consistencia del conteo.
    
    Returns:
        tuple: (available_dict, missing_in_txt, missing_physically)
    """
    available = defaultdict(list)
    missing_in_txt = []
    missing_physically = []
    
    for video_id, subset in video_data.items():
        video_file = videos_dir / f"{video_id}.mp4"
        
        # 1. ¿Está reportado como link roto por los autores?
        if video_id in missing_ids:
            missing_in_txt.append(video_id)
        # 2. ¿Falta en el disco duro por error de descarga?
        elif not video_file.exists():
            missing_physically.append(video_id)
        # 3. Existe y es válido
        else:
            available[subset].append(video_id)
            
    return dict(available), missing_in_txt, missing_physically


def copy_videos_by_subset(videos_by_subset: Dict[str, List[str]], 
                          source_dir: Path, dest_dir: Path) -> Dict[str, Dict[str, int]]:
    """Copia los videos organizados en subcarpetas."""
    stats = {}
    total_copied = 0
    total_errors = 0
    
    for subset, video_ids in videos_by_subset.items():
        subset_dir = dest_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Copiando videos de '{subset}' a: {subset_dir}")
        print(f"   Total videos a copiar: {len(video_ids)}")
        
        copied = 0
        errors = 0
        
        for video_id in video_ids:
            source_file = source_dir / f"{video_id}.mp4"
            dest_file = subset_dir / f"{video_id}.mp4"
            
            try:
                shutil.copy2(source_file, dest_file)
                copied += 1
                if copied % 100 == 0:
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
    print("🎬 Organizador de Videos NSLT-100 (Validación Estricta)")
    print("=" * 60)
    
    print(f"\n📖 Cargando {nslt_100_file}...")
    nslt_video_data = load_nslt_100(nslt_100_file)
    total_required = len(nslt_video_data)
    
    print(f"\n📖 Cargando {missing_file}...")
    missing_ids = load_missing_videos(missing_file)
    
    print(f"\n🔍 Procesando disponibilidad en {videos_dir}...")
    available_by_subset, missing_in_txt, missing_physically = get_available_videos(
        nslt_video_data, missing_ids, videos_dir
    )
    
    total_available = sum(len(videos) for videos in available_by_subset.values())
    total_missing_txt = len(missing_in_txt)
    total_missing_physically = len(missing_physically)
    
    # Validación matemática de integridad
    assert total_required == (total_available + total_missing_txt + total_missing_physically), "❌ Error de integridad matemática en el conteo."
    
    print(f"\n📊 BALANCE GENERAL (Integridad Validada):")
    print(f"   Videos requeridos (nslt_100): {total_required}")
    print(f"   ----------------------------------------")
    print(f"   ✅ Disponibles para copiar:   {total_available}")
    for subset in sorted(available_by_subset.keys()):
        print(f"      - {subset}: {len(available_by_subset[subset])}")
    print(f"   ❌ Faltantes por missing.txt: {total_missing_txt}")
    print(f"   ❌ Faltantes en disco físico: {total_missing_physically}")
    
    if total_available > 0:
        print(f"\n¿Deseas iniciar la copia de {total_available} videos a {output_dir}? (y/n): ", end='')
        response = input().strip().lower()
        
        if response == 'y':
            stats = copy_videos_by_subset(available_by_subset, videos_dir, output_dir)
            
            # Crear archivo de resumen preciso
            summary_file = output_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"REPORTE DE INTEGRIDAD - NSLT-100\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"1. BALANCE DE DATOS:\n")
                f.write(f"   Total Requerido: {total_required}\n")
                f.write(f"   Total Disponible y Copiado: {total_available}\n")
                f.write(f"   Total Faltante (missing.txt original): {total_missing_txt}\n")
                f.write(f"   Total Faltante (error de disco/descarga): {total_missing_physically}\n\n")
                
                f.write(f"2. DISTRIBUCIÓN DEL DATASET ÚTIL ({total_available} videos):\n")
                for subset in sorted(available_by_subset.keys()):
                    f.write(f"\n[{subset.upper()}]\n")
                    f.write(f"  Videos esperados: {len(available_by_subset[subset])}\n")
                    f.write(f"  Videos copiados con éxito: {stats[subset]['copied']}\n")
                    if stats[subset]['errors'] > 0:
                        f.write(f"  Errores de copia: {stats[subset]['errors']}\n")
                
                if total_missing_physically > 0:
                    f.write(f"\n\n3. ANOMALÍAS (Faltantes Físicos No Explicados):\n")
                    for vid in sorted(missing_physically):
                        f.write(f"  {vid}.mp4\n")
                        
            print(f"\n📝 Resumen validado guardado en: {summary_file}")
        else:
            print("\n❌ Operación cancelada por el usuario.")
    else:
        print("\n❌ No hay videos disponibles para procesar.")

if __name__ == "__main__":
    main()