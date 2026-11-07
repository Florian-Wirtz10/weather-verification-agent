import type { CaseOption, DataMetadata, Initialization, Job, Scope, TaskPayload } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    const detail = body.detail
    const message = Array.isArray(detail)
      ? detail.map((item: { loc?: (string | number)[]; msg?: string }) =>
        `${item.loc?.join('.') || '参数'}: ${item.msg || '无效'}`).join('；')
      : typeof detail === 'string' ? detail : '请求失败'
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),
  options: () => request<{ cases: CaseOption[] }>('/api/catalog/options'),
  metadata: () => request<DataMetadata>('/api/catalog/metadata'),
  initializations: (caseId: string, models: string[], completeOnly: boolean) =>
    request<{ initializations: Initialization[] }>('/api/catalog/initializations', {
      method: 'POST',
      body: JSON.stringify({ case_id: caseId, models, complete_only: completeOnly }),
    }),
  commonTimes: (caseId: string, models: string[], ids: string[], completeOnly: boolean) =>
    request<{ valid_times: { value: string; label: string }[] }>(
      '/api/catalog/common-valid-times',
      {
        method: 'POST',
        body: JSON.stringify({
          case_id: caseId,
          models,
          initialization_ids: ids,
          complete_only: completeOnly,
        }),
      },
    ),
  defaultPrompt: (caseId: string, models: string[], scope: Scope, completeOnly: boolean) =>
    request<{ prompt: string }>('/api/tasks/default-prompt', {
      method: 'POST',
      body: JSON.stringify({
        case_id: caseId,
        models,
        scope,
        complete_only: completeOnly,
      }),
    }),
  createJob: (payload: TaskPayload) =>
    request<Job>('/api/jobs', { method: 'POST', body: JSON.stringify(payload) }),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  jobs: () => request<{ jobs: Job[] }>('/api/jobs'),
}
