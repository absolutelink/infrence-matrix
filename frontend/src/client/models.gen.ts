import { client } from "./client.gen"

export interface Model {
  id: string
  name: string
  path: string
  size_bytes: number
  architecture: string
  parameter_count: number | null
  quantization: string
  supports_embeddings: boolean
  supports_vision: boolean
  context_length: number
  license: string | null
  tags: string[]
  description: string | null
  source: string
  source_repo_id: string | null
  source_url: string | null
  source_file: string | null
  downloaded_at: string
  updated_at: string | null
}

export interface ModelsServiceReadModelsData {
  query?: {
    skip?: number
    limit?: number
  }
}

export interface ModelsServiceReadModelData {
  id: string
}

export interface ModelsServiceCreateModelData {
  body: {
    name: string
    path: string
    size_bytes: number
    architecture: string
    parameter_count?: number | null
    quantization: string
    supports_embeddings?: boolean
    supports_vision?: boolean
    context_length: number
    license?: string | null
    tags?: string[]
    description?: string | null
    source: string
    source_repo_id?: string | null
    source_url?: string | null
    source_file?: string | null
  }
}

export interface ModelsServiceUpdateModelData {
  id: string
  body: {
    name?: string | null
    path?: string | null
    size_bytes?: number | null
    architecture?: string | null
    parameter_count?: number | null
    quantization?: string | null
    supports_embeddings?: boolean | null
    supports_vision?: boolean | null
    context_length?: number | null
    license?: string | null
    tags?: string[] | null
    description?: string | null
    source?: string | null
    source_repo_id?: string | null
    source_url?: string | null
    source_file?: string | null
  }
}

export interface ModelsServiceDeleteModelData {
  id: string
}

export const ModelsService = {
  readModels: async (
    data: ModelsServiceReadModelsData = {},
  ): Promise<{ data: Model[] }> => {
    const response = await client.get({
      url: "/api/v1/models/",
      security: [{ scheme: "bearer", type: "http" }],
      ...data,
    })
    return { data: response.data as Model[] }
  },

  readModel: async (
    data: ModelsServiceReadModelData,
  ): Promise<{ data: Model }> => {
    const response = await client.get({
      url: `/api/v1/models/${data.id}`,
      security: [{ scheme: "bearer", type: "http" }],
    })
    return { data: response.data as Model }
  },

  createModel: async (
    data: ModelsServiceCreateModelData,
  ): Promise<{ data: Model }> => {
    const response = await client.post({
      url: "/api/v1/models/",
      security: [{ scheme: "bearer", type: "http" }],
      body: data.body,
    })
    return { data: response.data as Model }
  },

  updateModel: async (
    data: ModelsServiceUpdateModelData,
  ): Promise<{ data: Model }> => {
    const response = await client.put({
      url: `/api/v1/models/${data.id}`,
      security: [{ scheme: "bearer", type: "http" }],
      body: data.body,
    })
    return { data: response.data as Model }
  },

  deleteModel: async (
    data: ModelsServiceDeleteModelData,
  ): Promise<{ data: { message: string } }> => {
    const response = await client.delete({
      url: `/api/v1/models/${data.id}`,
      security: [{ scheme: "bearer", type: "http" }],
    })
    return { data: response.data as { message: string } }
  },
}
