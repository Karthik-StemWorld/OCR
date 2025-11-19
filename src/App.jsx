import React, { useState, useEffect, useRef } from 'react';
import { Download, Eye, FileText, Table2, CheckCircle, AlertCircle, X, Upload, MapPin, RotateCw, RotateCcw, CheckSquare, ZoomIn, ZoomOut } from 'lucide-react';

export default function EnhancedTableOCRSystem() {
  const [files, setFiles] = useState([]);
  const [processing, setProcessing] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  const [results, setResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [previewFile, setPreviewFile] = useState(null);
  const [previewZoom, setPreviewZoom] = useState(1);
  const [resultZoom, setResultZoom] = useState(1);
  const [uploadStats, setUploadStats] = useState({ total: 0, validated: 0, pending: 0 });
  const [showDuplicateModal, setShowDuplicateModal] = useState(false);
  const [duplicateFiles, setDuplicateFiles] = useState([]);
  const [showSuccessModal, setShowSuccessModal] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [activeTab, setActiveTab] = useState('upload');
  
  const [districts, setDistricts] = useState([]);
  const [mandals, setMandals] = useState([]);
  const [villages, setVillages] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState("");
  const [selectedMandal, setSelectedMandal] = useState("");
  const [selectedVillage, setSelectedVillage] = useState("");
  const [selectedMonth, setSelectedMonth] = useState("");
  const [selectedYear, setSelectedYear] = useState("");

  const fileInputRef = useRef(null);

  const API_BASE = 'api.stemverse.app/OCR/api';
  const MAX_FILE_SIZE = 16 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'pdf', 'tiff', 'tif', 'bmp', 'webp'];

  // Auto-load CSV file on component mount
  useEffect(() => {
    fetchLocations();
    const interval = setInterval(() => {
      fetchLocations();
    }, 300000);
    
    return () => clearInterval(interval);
  }, []);

  async function fetchLocations() {
    try {
      const res = await fetch("/OCR/districts-mandals.csv?t=" + Date.now());
      if (!res.ok) throw new Error("Failed to load CSV");
      const csvText = await res.text();
      const lines = csvText.split('\n').filter(line => line.trim());
      await loadCSVDataFromText(lines);
    } catch (err) {
      console.warn("Could not auto-load CSV file:", err.message);
    }
  }

  const loadCSVDataFromText = async (lines) => {
    try {
      if (lines.length < 2) return;

      const headerValues = [];
      let current = '';
      let inQuotes = false;
      for (let char of lines[0]) {
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          headerValues.push(current.trim().replace(/^"|"$/g, '').toLowerCase());
          current = '';
        } else {
          current += char;
        }
      }
      headerValues.push(current.trim().replace(/^"|"$/g, '').toLowerCase());
      console.log("HEADER VALUES =>", headerValues);
      
      const mandalIdx = headerValues.findIndex(h => h.includes('mandals'));
      const districtIdx = headerValues.findIndex(h => h.includes('districts'));
      const villageIdx = headerValues.findIndex(h => h.includes('villages'));
      
      if (districtIdx === -1 || mandalIdx === -1) {
        console.error('CSV must contain mandals and districts columns');
        return;
      }

      const districtMap = new Map();
      
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const values = [];
        let current = '';
        let inQuotes = false;
        
        for (let char of line) {
          if (char === '"') {
            inQuotes = !inQuotes;
          } else if (char === ',' && !inQuotes) {
            values.push(current.trim().replace(/^"|"$/g, ''));
            current = '';
          } else {
            current += char;
          }
        }
        values.push(current.trim().replace(/^"|"$/g, ''));
        
        const mandalName = values[mandalIdx]?.trim();
        const districtName = values[districtIdx]?.trim();
        const villageName = villageIdx !== -1 ? values[villageIdx]?.trim() : '';
        
        if (!districtName || !mandalName) continue;
        
        if (!districtMap.has(districtName)) {
          districtMap.set(districtName, {
            id: `d_${districtName.toLowerCase().replace(/\s+/g, '_').replace(/\./g, '').replace(/[^a-z0-9_]/g, '')}`,
            name: districtName,
            mandals: new Map()
          });
        }
        
        const district = districtMap.get(districtName);
        const mandalKey = mandalName.toLowerCase();
        
        if (!district.mandals.has(mandalKey)) {
          district.mandals.set(mandalKey, {
            id: `m_${mandalName.toLowerCase().replace(/\s+/g, '_').replace(/\./g, '').replace(/[^a-z0-9_]/g, '')}`,
            name: mandalName,
            villages: []
          });
        }
        
        const mandal = district.mandals.get(mandalKey);
        
        if (villageName) {
          const villageId = `v_${villageName.toLowerCase().replace(/\s+/g, '_').replace(/\./g, '').replace(/[^a-z0-9_]/g, '')}`;
          if (!mandal.villages.some(v => v.id === villageId)) {
            mandal.villages.push({
              id: villageId,
              name: villageName
            });
          }
        } else if (mandal.villages.length === 0) {
          // If no villages specified, add mandal name as default village
          const defaultVillageId = `v_${mandalName.toLowerCase().replace(/\s+/g, '_').replace(/\./g, '').replace(/[^a-z0-9_]/g, '')}`;
          if (!mandal.villages.some(v => v.id === defaultVillageId)) {
            mandal.villages.push({
              id: defaultVillageId,
              name: mandalName
            });
          }
        }
      }
      
      const districtsArray = Array.from(districtMap.values()).map(d => ({
        ...d,
        mandals: Array.from(d.mandals.values())
          .map(m => ({
            ...m,
            villages: m.villages.sort((a, b) => a.name.localeCompare(b.name))
          }))
          .sort((a, b) => a.name.localeCompare(b.name))
      })).sort((a, b) => a.name.localeCompare(b.name));
      
      setDistricts(districtsArray);
      
      const totalMandals = districtsArray.reduce((sum, d) => sum + d.mandals.length, 0);
      const totalVillages = districtsArray.reduce((sum, d) => 
        sum + d.mandals.reduce((mSum, m) => mSum + m.villages.length, 0), 0
      );
      
      console.log('✅ CSV Data loaded successfully:');
      console.log(`📊 Total Districts: ${districtsArray.length}`);
      console.log(`📊 Total Mandals: ${totalMandals}`);
      console.log(`📊 Total Villages: ${totalVillages}`);
      
      // Debug: Show first few entries
      if (districtsArray.length > 0) {
        console.log('\n📋 Sample Data:');
        districtsArray.slice(0, 2).forEach(d => {
          console.log(`\n  District: ${d.name}`);
          d.mandals.slice(0, 3).forEach(m => {
            console.log(`    - Mandal: ${m.name}`);
            m.villages.slice(0, 3).forEach(v => {
              console.log(`      • Village: ${v.name}`);
            });
            if (m.villages.length > 3) {
              console.log(`      ... and ${m.villages.length - 3} more villages`);
            }
          });
        });
      }
    } catch (error) {
      console.error('Error loading CSV:', error);
    }
  };

  useEffect(() => {
    const validated = files.filter(f => f.validated).length;
    const pending = files.length - validated;
    setUploadStats({ total: files.length, validated, pending });
  }, [files]);

  useEffect(() => {
    if (!selectedDistrict) {
      setMandals([]);
      setVillages([]);
      setSelectedMandal("");
      setSelectedVillage("");
      return;
    }
    const district = districts.find(d => d.id === selectedDistrict);
    if (district) {
      setMandals(district.mandals || []);
      setSelectedMandal("");
      setSelectedVillage("");
      setVillages([]);
    }
  }, [selectedDistrict, districts]);

  useEffect(() => {
    if (!selectedMandal) {
      setVillages([]);
      setSelectedVillage("");
      return;
    }
    const mandal = mandals.find(m => m.id === selectedMandal);
    if (mandal) {
      setVillages(mandal.villages || []);
      setSelectedVillage("");
    }
  }, [selectedMandal, mandals]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (selectedFiles.length === 0) return;

    const duplicates = [];
    const invalidFiles = [];
    const oversizedFiles = [];
    const validNewFiles = [];

    selectedFiles.forEach(file => {
      const ext = file.name.split('.').pop()?.toLowerCase();
      
      if (!ext || !ALLOWED_EXTENSIONS.includes(ext)) {
        invalidFiles.push(file.name);
        return;
      }

      if (file.size > MAX_FILE_SIZE) {
        oversizedFiles.push(file.name);
        return;
      }

      const isDuplicate = files.some(f => 
        f.name === file.name && f.size === file.size
      );

      if (isDuplicate) {
        duplicates.push(file.name);
      } else {
        validNewFiles.push(file);
      }
    });

    const newFiles = validNewFiles.map(file => ({
      id: Date.now() + Math.random(),
      file: file,
      name: file.name,
      size: file.size,
      type: file.type,
      validated: false,
      previewUrl: null,
      rotation: 0
    }));

    let messages = [];
    if (newFiles.length > 0) {
      messages.push(`✅ ${newFiles.length} file(s) added successfully`);
    }
    if (duplicates.length > 0) {
      setDuplicateFiles(duplicates);
      setShowDuplicateModal(true);
    }
    if (invalidFiles.length > 0) {
      messages.push(`❌ ${invalidFiles.length} invalid file type(s)`);
    }
    if (oversizedFiles.length > 0) {
      messages.push(`❌ ${oversizedFiles.length} file(s) too large (max 16MB)`);
    }

    if (messages.length > 0) {
      setMessage(messages.join('\n'));
      setMessageType(newFiles.length > 0 ? 'info' : 'error');
    }

    if (newFiles.length > 0) {
      setFiles([...files, ...newFiles]);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const droppedFiles = Array.from(e.dataTransfer.files);
    const fakeEvent = { target: { files: droppedFiles } };
    handleFileChange(fakeEvent);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const removeFile = (fileId) => {
    const fileToRemove = files.find(f => f.id === fileId);
    if (fileToRemove?.previewUrl) {
      URL.revokeObjectURL(fileToRemove.previewUrl);
    }
    setFiles(files.filter(f => f.id !== fileId));
  };

  const validateFile = (fileId) => {
    setFiles(files.map(f => 
      f.id === fileId ? { ...f, validated: true } : f
    ));
    setMessage('✅ File validated successfully');
    setMessageType('success');
  };

  const validateAllFiles = () => {
    if (files.length === 0) {
      setMessage('❌ No files to validate');
      setMessageType('error');
      return;
    }
    setFiles(files.map(f => ({ ...f, validated: true })));
    setMessage(`✅ All ${files.length} file(s) validated successfully!`);
    setMessageType('success');
  };

  const previewFileHandler = (fileObj) => {
    if (!fileObj.previewUrl && fileObj.file) {
      fileObj.previewUrl = URL.createObjectURL(fileObj.file);
    }
    if (fileObj.rotation === undefined) {
      fileObj.rotation = 0;
    }
    setPreviewFile({ ...fileObj });
  };

  const closePreview = () => {
    setPreviewFile(null);
    setPreviewZoom(1);
  };

  const rotateImage = (direction) => {
    if (!previewFile) return;
    
    const rotationIncrement = direction === 'right' ? 90 : -90;
    const newRotation = ((previewFile.rotation || 0) + rotationIncrement) % 360;
    
    const updatedFile = { ...previewFile, rotation: newRotation };
    setPreviewFile(updatedFile);
    
    setFiles(files.map(f => 
      f.id === previewFile.id ? { ...f, rotation: newRotation } : f
    ));
  };

  const rotateImageFile = async (file, rotation) => {
    if (!rotation || rotation === 0 || rotation % 360 === 0) {
      return file;
    }

    try {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          img.onload = () => {
            if (rotation === 90 || rotation === 270 || rotation === -90 || rotation === -270) {
              canvas.width = img.height;
              canvas.height = img.width;
            } else {
              canvas.width = img.width;
              canvas.height = img.height;
            }
            
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.rotate((rotation * Math.PI) / 180);
            ctx.drawImage(img, -img.width / 2, -img.height / 2);
            
            canvas.toBlob((blob) => {
              if (blob) {
                const rotatedFile = new File([blob], file.name, {
                  type: file.type || 'image/jpeg',
                  lastModified: Date.now()
                });
                resolve(rotatedFile);
              } else {
                reject(new Error('Failed to create rotated image'));
              }
            }, file.type || 'image/jpeg', 0.95);
          };
          img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    } catch (error) {
      console.error('Error rotating image:', error);
      return file;
    }
  };

  const handleProcess = async () => {
    if (files.length === 0) {
      setMessage('❌ Please select at least one file');
      setMessageType('error');
      return;
    }

    const allValidated = files.every(f => f.validated);
    if (!allValidated) {
      setMessage('❌ Please validate all files before processing');
      setMessageType('error');
      return;
    }

    if (!selectedDistrict || !selectedMandal || !selectedVillage) {
      setMessage('❌ Please select District, Mandal, and Village');
      setMessageType('error');
      return;
    }

    if (!selectedMonth || !selectedYear) {
      setMessage('❌ Please select Month and Year');
      setMessageType('error');
      return;
    }

    try {
      setProcessing(true);
      setMessage('🔍 Extracting tables from images...');
      setMessageType('info');

      const allResults = [];
      const districtName = districts.find(d => d.id === selectedDistrict)?.name;
      const mandalName = mandals.find(m => m.id === selectedMandal)?.name;
      const villageName = villages.find(v => v.id === selectedVillage)?.name;
      const monthName = new Date(2000, parseInt(selectedMonth) - 1).toLocaleString('default', { month: 'long' });

      let totalProcessed = 0;
      let successCount = 0;
      let errorCount = 0;

      for (const fileObj of files) {
        totalProcessed++;
        setMessage(
          `🔬 Processing ${totalProcessed}/${files.length}: ${fileObj.name}\n` +
          `📐 Detecting table structure...`
        );

        let fileToProcess = fileObj.file;
        if (fileObj.rotation && fileObj.rotation !== 0) {
          try {
            fileToProcess = await rotateImageFile(fileObj.file, fileObj.rotation);
            setMessage(
              `🔄 Rotating image (${fileObj.rotation}°)...\n` +
              `🔬 Processing ${totalProcessed}/${files.length}: ${fileObj.name}`
            );
          } catch (err) {
            console.warn('Could not rotate image, using original:', err);
          }
        }

        const formData = new FormData();
        formData.append('file', fileToProcess);

        try {
          const response = await fetch(`${API_BASE}/api/extract-tables`, {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
          }

          const result = await response.json();
          
          if (result.success && result.tables && result.tables.length > 0) {
            result.tables.forEach((table, idx) => {
              if (table.dataframe && table.dataframe.length > 0) {
                allResults.push({
                  id: Date.now() + idx + Math.random(),
                  filename: fileObj.name,
                  mode: 'table',
                  tableNumber: idx + 1,
                  totalTables: result.tables.length,
                  data: table.dataframe,
                  rowCount: table.row_count,
                  colCount: table.col_count,
                  csvData: table.csv,
                  jsonData: table.json,
                  htmlData: table.html,
                  headers: table.headers || Object.keys(table.dataframe[0] || {}),
                  district: districtName,
                  mandal: mandalName,
                  village: villageName,
                  month: monthName,
                  year: selectedYear,
                  ocrEngine: 'google-vision-api',
                  extractionMethod: 'vision-api-structure',
                  timestamp: new Date().toLocaleString()
                });
              }
            });
            successCount++;
          } else {
            errorCount++;
            console.error('No tables found in response:', result);
          }
        } catch (err) {
          console.error('Table extraction failed for', fileObj.name, err);
          errorCount++;
          setMessage(
            `❌ Error processing ${fileObj.name}\n` +
            `Server Error: ${err.message}`
          );
          setMessageType('error');
          setProcessing(false);
          return;
        }
      }

      if (allResults.length > 0) {
        setResults([...results, ...allResults]);
        
        const totalRows = allResults.reduce((sum, r) => sum + (r.rowCount || 0), 0);
        
        const successMsg = `✅ OCR Conversion Completed Successfully!\n\n` +
          `📊 Total Tables Extracted: ${allResults.length}\n` +
          `📋 Total Rows: ${totalRows}\n` +
          `📋 Total Columns: ${allResults[0]?.colCount || 0}\n` +
          `✅ Success: ${successCount} file(s)\n` +
          `${errorCount > 0 ? `❌ Errors: ${errorCount} file(s)\n` : ''}` +
          `📍 Location: ${districtName} → ${mandalName} → ${villageName}`;
        
        setSuccessMessage(successMsg);
        setShowSuccessModal(true);
        setActiveTab('results');
      } else {
        setMessage(
          `⚠️ Processing completed but no tables extracted\n` +
          `📁 Processed: ${totalProcessed} file(s)\n` +
          `❌ No valid table data found`
        );
        setMessageType('error');
      }

      files.forEach(f => {
        if (f.previewUrl) URL.revokeObjectURL(f.previewUrl);
      });
      setFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      console.error('Processing error:', err);
      setMessage(
        `❌ Connection Error\n\n` +
        `Cannot connect to backend server at ${API_BASE}\n\n` +
        `Error: ${err.message}`
      );
      setMessageType('error');
    } finally {
      setProcessing(false);
    }
  };

  const exportAsCSV = (result) => {
    const csvContent = result.csvData || '';
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    
    const filenameBase = result.filename.split('.')[0];
    const monthYearSuffix = result.month && result.year ? `_${result.month}_${result.year}` : '';
    link.download = `${filenameBase}_table_${result.tableNumber}${monthYearSuffix}_export.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const viewResult = (result) => {
    setSelectedResult(result);
  };

  const closeModal = () => {
    setSelectedResult(null);
    setResultZoom(1);
  };

  const formatBytes = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="fixed inset-0 bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 overflow-hidden">
      <div className="h-full overflow-y-auto">
        <div className="max-w-7xl mx-auto p-4 lg:p-6">
          {/* Header */}
          <div className="text-center mb-6">
            <h2 className="text-2xl lg:text-4xl font-extrabold mb-2 bg-gradient-to-r from-yellow-300 via-orange-300 to-yellow-300 bg-clip-text text-transparent">
              SOCIETY FOR ELIMINATION OF RURAL POVERTY
            </h2>
            <h3 className="text-lg lg:text-2xl font-semibold text-blue-200">
              Department of Rural Development, Government of Andhra Pradesh
            </h3>
          </div>
          
          {/* Main Header */}
          <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-6 lg:p-8 mb-6 shadow-2xl border border-white/20">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-12 h-12 lg:w-14 lg:h-14 bg-gradient-to-br from-yellow-400 to-orange-600 rounded-xl flex items-center justify-center shadow-lg">
                <Table2 size={28} className="text-white" />
              </div>
              <div className="flex-1">
                <h1 className="text-xl lg:text-2xl font-extrabold text-white mb-2">
                  Digitalizing SHG Data
                </h1>
              </div>
            </div>

            {/* Tab Navigation */}
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => setActiveTab('upload')}
                className={`px-6 py-3 rounded-xl font-bold transition-all flex items-center gap-2 ${
                  activeTab === 'upload'
                    ? 'bg-white text-purple-900 shadow-lg'
                    : 'bg-white/20 text-white hover:bg-white/30'
                }`}
              >
                <Upload size={20} />
                Upload & Process
              </button>
              <button
                onClick={() => setActiveTab('results')}
                className={`px-6 py-3 rounded-xl font-bold transition-all flex items-center gap-2 ${
                  activeTab === 'results'
                    ? 'bg-white text-purple-900 shadow-lg'
                    : 'bg-white/20 text-white hover:bg-white/30'
                }`}
              >
                <Table2 size={20} />
                Converted Results {results.length > 0 && `(${results.length})`}
              </button>
            </div>
          </div>

          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <>
              {/* Location Selection */}
              <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-2xl p-6 border-2 border-red-300 mb-6 shadow-lg">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 bg-gradient-to-br from-red-500 to-orange-500 rounded-xl flex items-center justify-center shadow-md">
                    <MapPin size={28} className="text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl lg:text-2xl font-bold text-red-900">
                      Location Selection <span className="text-red-600">*</span>
                    </h3>
                    <p className="text-sm text-red-700">Select District, Mandal, and Village</p>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="relative">
                    <label className="block text-sm font-bold text-red-900 mb-2">
                      District <span className="text-red-600">*</span>
                    </label>
                    <select
                      value={selectedDistrict}
                      onChange={(e) => setSelectedDistrict(e.target.value)}
                      className="w-full px-4 py-3 text-base font-medium border-2 border-red-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-red-500 bg-white appearance-none cursor-pointer"
                    >
                      <option value="">-- Select District --</option>
                      {districts.map((d) => (
                        <option key={d.id} value={d.id}>{d.name}</option>
                      ))}
                    </select>
                  </div>

                  <div className="relative">
                    <label className="block text-sm font-bold text-red-900 mb-2">
                      Mandal <span className="text-red-600">*</span>
                    </label>
                    <select
                      value={selectedMandal}
                      onChange={(e) => setSelectedMandal(e.target.value)}
                      disabled={!mandals.length}
                      className="w-full px-4 py-3 text-base font-medium border-2 border-red-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 bg-white appearance-none cursor-pointer disabled:bg-gray-100 disabled:cursor-not-allowed"
                    >
                      <option value="">-- Select Mandal --</option>
                      {mandals.map((m) => (
                        <option key={m.id} value={m.id}>{m.name}</option>
                      ))}
                    </select>
                  </div>

                  <div className="relative">
                    <label className="block text-sm font-bold text-red-900 mb-2">
                      Village <span className="text-red-600">*</span>
                    </label>
                    <select
                      value={selectedVillage}
                      onChange={(e) => setSelectedVillage(e.target.value)}
                      disabled={!villages.length}
                      className="w-full px-4 py-3 text-base font-medium border-2 border-red-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-red-500 bg-white appearance-none cursor-pointer disabled:bg-gray-100 disabled:cursor-not-allowed"
                    >
                      <option value="">-- Select Village --</option>
                      {villages.map((v) => (
                        <option key={v.id} value={v.id}>{v.name}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Month/Year and Upload Section */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                {/* Month and Year */}
                <div className="lg:col-span-1 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-6 border-2 border-blue-300 shadow-lg">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center shadow-md">
                      <FileText size={28} className="text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg lg:text-xl font-bold text-blue-900">
                        Month & Year <span className="text-blue-600">*</span>
                      </h3>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-bold text-blue-900 mb-2">
                        Month <span className="text-blue-600">*</span>
                      </label>
                      <select
                        value={selectedMonth}
                        onChange={(e) => setSelectedMonth(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white appearance-none cursor-pointer"
                      >
                        <option value="">-- Select Month --</option>
                        <option value="01">January</option>
                        <option value="02">February</option>
                        <option value="03">March</option>
                        <option value="04">April</option>
                        <option value="05">May</option>
                        <option value="06">June</option>
                        <option value="07">July</option>
                        <option value="08">August</option>
                        <option value="09">September</option>
                        <option value="10">October</option>
                        <option value="11">November</option>
                        <option value="12">December</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-bold text-blue-900 mb-2">
                        Year <span className="text-blue-600">*</span>
                      </label>
                      <select
                        value={selectedYear}
                        onChange={(e) => setSelectedYear(e.target.value)}
                        className="w-full px-4 py-3 border-2 border-blue-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white appearance-none cursor-pointer"
                      >
                        <option value="">-- Select Year --</option>
                        {Array.from({ length: 10 }, (_, i) => {
                          const year = new Date().getFullYear() - i;
                          return <option key={year} value={year}>{year}</option>;
                        })}
                      </select>
                    </div>
                  </div>
                </div>

                {/* File Upload Section */}
                <div className="lg:col-span-2 bg-white rounded-3xl shadow-2xl p-6 lg:p-8">
                  <h2 className="text-2xl lg:text-3xl font-bold text-gray-800 flex items-center gap-2 mb-6">
                    <Upload size={32} className="text-indigo-600" />
                    Upload Files
                  </h2>

                  {files.length > 0 && (
                    <div className="grid grid-cols-3 gap-4 mb-6">
                      <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-blue-800">{uploadStats.total}</div>
                        <div className="text-xs text-blue-600 font-semibold">Total Files</div>
                      </div>
                      <div className="bg-green-50 border-2 border-green-300 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-green-800">{uploadStats.validated}</div>
                        <div className="text-xs text-green-600 font-semibold">Validated</div>
                      </div>
                      <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-3 text-center">
                        <div className="text-2xl font-bold text-orange-800">{uploadStats.pending}</div>
                        <div className="text-xs text-orange-600 font-semibold">Pending</div>
                      </div>
                    </div>
                  )}

                  <div
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    className="border-4 border-dashed border-indigo-300 rounded-2xl p-8 lg:p-12 bg-gradient-to-br from-indigo-50 to-purple-50 hover:from-indigo-100 hover:to-purple-100 transition-all cursor-pointer"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <div className="text-center">
                      <Upload size={64} className="mx-auto text-indigo-600 mb-4" />
                      <p className="text-xl font-bold text-gray-800 mb-2">
                        Drop files here or click to upload
                      </p>
                      <p className="text-sm text-gray-600">
                        Supports: PNG, JPG, PDF, TIFF (Max 16MB)
                      </p>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept=".png,.jpg,.jpeg,.pdf,.tiff,.tif,.bmp,.webp"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </div>

                  {files.length > 0 && (
                    <div className="mt-6 mb-4 flex justify-end">
                      <button
                        onClick={validateAllFiles}
                        className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-all font-semibold flex items-center gap-2 shadow-md"
                      >
                        <CheckCircle size={20} />
                        Validate All Documents
                      </button>
                    </div>
                  )}

                  {files.length > 0 && (
                    <div className="mt-6 space-y-3">
                      <h3 className="text-lg font-bold text-gray-800">Selected Files ({files.length})</h3>
                      {files.map(fileObj => (
                        <div key={fileObj.id} className="bg-gray-50 border-2 border-gray-300 rounded-lg p-4 flex items-center gap-4">
                          <FileText size={32} className="text-indigo-600 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="font-semibold text-gray-800 truncate">{fileObj.name}</p>
                            <p className="text-sm text-gray-600">{formatBytes(fileObj.size)}</p>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            {fileObj.validated ? (
                              <span className="px-3 py-1 bg-green-500 text-white text-xs rounded-full font-bold flex items-center gap-1">
                                <CheckCircle size={14} />
                                Validated
                              </span>
                            ) : (
                              <button
                                onClick={() => validateFile(fileObj.id)}
                                className="px-3 py-1 bg-yellow-500 hover:bg-yellow-600 text-white text-xs rounded-full font-bold"
                              >
                                Validate
                              </button>
                            )}
                            <button
                              onClick={() => previewFileHandler(fileObj)}
                              className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg"
                            >
                              <Eye size={18} />
                            </button>
                            <button
                              onClick={() => removeFile(fileObj.id)}
                              className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
                            >
                              <X size={18} />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {message && (
                    <div className={`mt-6 p-4 rounded-lg border-2 ${
                      messageType === 'success' ? 'bg-green-50 border-green-300 text-green-800' :
                      messageType === 'error' ? 'bg-red-50 border-red-300 text-red-800' :
                      'bg-blue-50 border-blue-300 text-blue-800'
                    }`}>
                      <p className="whitespace-pre-line font-semibold">{message}</p>
                    </div>
                  )}

                  <div className="mt-6">
                    <button
                      onClick={handleProcess}
                      disabled={processing || files.length === 0}
                      className="w-full px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 disabled:from-gray-400 disabled:to-gray-500 text-white rounded-xl font-bold text-lg transition-all shadow-lg disabled:cursor-not-allowed flex items-center justify-center gap-3"
                    >
                      {processing ? (
                        <>
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                          Processing...
                        </>
                      ) : (
                        <>
                          <Table2 size={24} />
                          Converting into Digital File
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && (
            <div className="bg-white rounded-3xl shadow-2xl p-6 lg:p-8">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl lg:text-3xl font-bold text-gray-800 flex items-center gap-2">
                  <Table2 size={32} className="text-indigo-600" />
                  Converted Results ({results.length})
                </h2>
              </div>

              {results.length === 0 ? (
                <div className="text-center py-12">
                  <Table2 size={64} className="mx-auto text-gray-400 mb-4" />
                  <p className="text-xl text-gray-600 font-semibold">No results yet</p>
                  <p className="text-gray-500 mt-2">Upload and process files to see results here</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {results.map(result => (
                    <div key={result.id} className="bg-gradient-to-br from-indigo-50 to-purple-50 border-2 border-indigo-300 rounded-xl p-4">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                          <h3 className="font-bold text-gray-800 text-lg mb-1">{result.filename}</h3>
                          <div className="flex flex-wrap gap-2 mb-2">
                            <span className="px-2 py-1 bg-indigo-500 text-white text-xs rounded-full font-bold">
                              Table {result.tableNumber}/{result.totalTables}
                            </span>
                          </div>
                          <div className="text-xs text-gray-600 space-y-1">
                            <p>📍 {result.district} → {result.mandal} → {result.village}</p>
                            {result.month && result.year && (
                              <p>📅 {result.month} {result.year}</p>
                            )}
                            <p>📊 {result.rowCount} rows × {result.colCount} cols</p>
                            <p>🕐 {result.timestamp}</p>
                          </div>
                        </div>
                      </div>

                      <div className="flex gap-2 mt-3">
                        <button
                          onClick={() => viewResult(result)}
                          className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold flex items-center justify-center gap-2"
                        >
                          <Eye size={18} />
                          View
                        </button>
                        <button
                          onClick={() => exportAsCSV(result)}
                          className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-semibold flex items-center justify-center gap-2"
                        >
                          <Download size={18} />
                          CSV
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Preview Modal */}
      {previewFile && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b-2 border-gray-300">
              <h3 className="text-xl font-bold text-gray-800">{previewFile.name}</h3>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => rotateImage('left')}
                    className="p-2 bg-white hover:bg-gray-200 text-gray-700 rounded transition-all"
                  >
                    <RotateCcw size={20} />
                  </button>
                  <button
                    onClick={() => rotateImage('right')}
                    className="p-2 bg-white hover:bg-gray-200 text-gray-700 rounded transition-all"
                  >
                    <RotateCw size={20} />
                  </button>
                </div>
                <button
                  onClick={closePreview}
                  className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
                >
                  <X size={24} />
                </button>
              </div>
            </div>
            <div className="p-4 overflow-auto max-h-[calc(90vh-80px)] flex items-center justify-center">
              {previewFile.previewUrl && (
                <img 
                  src={previewFile.previewUrl} 
                  alt={previewFile.name}
                  className="max-w-full h-auto mx-auto"
                  style={{ 
                    transform: `rotate(${previewFile.rotation || 0}deg) scale(${previewZoom})`,
                    maxHeight: 'calc(90vh - 120px)'
                  }}
                />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Result Detail Modal */}
      {selectedResult && (
        <div className="fixed inset-0 bg-black/80 z-50 flex flex-col">
          <div className="bg-white shadow-2xl flex flex-col h-full w-full">
            <div className="flex items-center justify-between p-6 border-b-2 border-gray-300">
              <div>
                <h3 className="text-2xl font-bold text-gray-800 truncate">{selectedResult.filename}</h3>
                <p className="text-sm text-gray-600 mt-2 flex items-center gap-2">
                  📍 {selectedResult.district} → {selectedResult.mandal} → {selectedResult.village}
                  {selectedResult.month && selectedResult.year && (
                    <span> | 📅 {selectedResult.month} {selectedResult.year}</span>
                  )}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => setResultZoom(prev => Math.max(prev - 0.25, 0.5))}
                    className="p-2 bg-white hover:bg-gray-200 text-gray-700 rounded"
                    disabled={resultZoom <= 0.5}
                  >
                    <ZoomOut size={20} />
                  </button>
                  <button
                    onClick={() => setResultZoom(1)}
                    className="px-3 py-2 bg-white hover:bg-gray-200 text-gray-700 rounded text-sm font-semibold"
                  >
                    {Math.round(resultZoom * 100)}%
                  </button>
                  <button
                    onClick={() => setResultZoom(prev => Math.min(prev + 0.25, 3))}
                    className="p-2 bg-white hover:bg-gray-200 text-gray-700 rounded"
                    disabled={resultZoom >= 3}
                  >
                    <ZoomIn size={20} />
                  </button>
                </div>
                <button
                  onClick={closeModal}
                  className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-lg"
                >
                  <X size={24} />
                </button>
              </div>
            </div>
            <div className="p-6 flex-1 overflow-auto result-modal-content" style={{ minHeight: 0, overflowX: 'auto', overflowY: 'auto' }}>
              <style>{`
                .result-modal-content .tables-wrapper {
                  max-height: none !important;
                  overflow-y: visible !important;
                  overflow-x: visible !important;
                  height: auto !important;
                  width: 100% !important;
                }
                .result-modal-content .table-section {
                  margin-bottom: 30px;
                }
                .result-modal-content .shg-table {
                  width: 100% !important;
                }
              `}</style>
              {selectedResult.htmlData ? (
                <div 
                  className="w-full"
                  style={{
                    transformOrigin: 'top left',
                    transform: `scale(${resultZoom})`,
                    minWidth: `${100 / resultZoom}%`
                  }}
                >
                  <div
                    style={{
                      width: '100%'
                    }}
                    dangerouslySetInnerHTML={{ __html: selectedResult.htmlData }}
                  />
                </div>
              ) : (
                <div 
                  className="w-full"
                  style={{
                    transformOrigin: 'top left',
                    transform: `scale(${resultZoom})`,
                    minWidth: `${100 / resultZoom}%`
                  }}
                >
                  <table className="w-full border-collapse border-2 border-gray-300">
                    <thead>
                      <tr className="bg-indigo-700 text-white">
                        {selectedResult.headers.map((header, idx) => (
                          <th key={idx} className="border-2 border-gray-300 px-2 py-2 text-center font-bold">
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {selectedResult.data.map((row, rowIdx) => (
                        <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                          {selectedResult.headers.map((header, cellIdx) => (
                            <td key={cellIdx} className="border-2 border-gray-300 px-2 py-2 text-sm">
                              {row[header] || ''}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Duplicate Modal */}
      {showDuplicateModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
                <AlertCircle size={32} className="text-orange-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-800">Duplicate Files!</h3>
            </div>
            <p className="text-gray-700 mb-4">The following files are already uploaded:</p>
            <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-4 mb-6 max-h-48 overflow-y-auto">
              {duplicateFiles.map((name, idx) => (
                <p key={idx} className="text-sm text-orange-800 font-semibold py-1">• {name}</p>
              ))}
            </div>
            <button
              onClick={() => {
                setShowDuplicateModal(false);
                setDuplicateFiles([]);
              }}
              className="w-full px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-lg font-bold"
            >
              OK, Got it
            </button>
          </div>
        </div>
      )}

      {/* Success Modal */}
      {showSuccessModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle size={40} className="text-green-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-800">Success!</h3>
                <p className="text-sm text-green-600 font-semibold">OCR conversion completed</p>
              </div>
            </div>
            <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4 mb-6">
              <p className="text-gray-800 whitespace-pre-line font-semibold text-sm">{successMessage}</p>
            </div>
            <button
              onClick={() => setShowSuccessModal(false)}
              className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-bold"
            >
              View Results
            </button>
          </div>
        </div>
      )}
    </div>
  );
}