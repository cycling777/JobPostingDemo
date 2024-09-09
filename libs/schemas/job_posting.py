from typing import Optional
from pydantic import BaseModel, Field

class JobPostingRequest(BaseModel):
    url: str
    company_name: str

class JobPostingResponse(BaseModel):
    company_name: str
    employee_count: str
    revenue: str
    establishment_date: str
    business_content: str
    industry: str
    company_address: str
    job_position_name: str
    employment_type: str
    probation_period: str
    job_description: str
    work_location: str
    application_qualifications: str
    selection_process: str
    upper_annual_income: int
    lower_annual_income: int
    salary_details: str
    holidays_vacations: str
    working_hours: str
    benefits_welfare: str
    passive_smoking_measures: str
    job_change_caution: str
    education: str
    age: str
    fee: str
    selection_measures: str
    desired_personality: str
    other_attractions: str
    recruiter_id: str
    
class JobPosting(BaseModel):
    company_name: str = Field(default="不明", title="企業名")
    employee_count: str = Field(default="不明", title="従業員数")
    revenue: str = Field(default="不明", title="売上高")
    establishment_date: str = Field(default="不明", title="会社設立日")
    business_content: str = Field(default="不明", title="事業内容")
    industry: str = Field(default="不明", title="業界")
    company_address: str = Field(default="不明", title="会社所在地")
    job_position_name: str = Field(default="不明", title="募集ポジション名")
    employment_type: str = Field(default="不明", title="雇用形態(期間)")
    probation_period: str = Field(default="不明", title="試用期間")
    job_description: str = Field(default="不明", title="業務内容")
    work_location: str = Field(default="不明", title="勤務地")
    application_qualifications: str = Field(default="不明", title="応募資格")
    selection_process: str = Field(default="不明", title="選考プロセス")
    upper_annual_income: str = Field(default="不明", title="年収上限[万円]")
    lower_annual_income: str = Field(default="不明", title="年収下限[万円]")
    salary_details: str = Field(default="不明", title="給与(詳細)")
    holidays_vacations: str = Field(default="不明", title="休日休暇")
    working_hours: str = Field(default="不明", title="勤務時間")
    benefits_welfare: str = Field(default="不明", title="待遇・福利厚生")
    passive_smoking_measures: str = Field(default="不明", title="受動喫煙防止措置")
    job_change_caution: str = Field(default="不明", title="転職回数")
    education: str = Field(default="不明", title="学歴")
    age: str = Field(default="不明", title="年齢")
    fee: str = Field(default="不明", title="フィー（紹介手数料）")
    selection_measures: str = Field(default="不明", title="選考対策")
    desired_personality: str = Field(default="不明", title="求める人物像")
    other_attractions: str = Field(default="不明", title="その他魅力")
    recruiter_id: str = Field(default="不明", title="採用担当者ID")